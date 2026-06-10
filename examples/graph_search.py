"""
graph_search.py — Graph-augmented retrieval with Quanta.

Demonstrates how combining dense vector search with a Neo4j knowledge graph
surfaces documents that pure ANN search would rank low or miss entirely.

Use case: a small Greek legal document corpus covering GDPR / data protection.
Five documents are connected by legal citation, interpretation, and amendment
relationships.  A query about "controller obligations" naturally scores the
core GDPR statutes highly.  Graph traversal then promotes related court
decisions and circulars that dense search alone would rank poorly.

Run:
    python examples/graph_search.py

Required .env variables:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    DOCSTORE_BACKEND=duckdb
    DUCKDB_PATH=./examples/graph_search_example.duckdb

    # QuantaSettings validates POSTGRES_* even for DuckDB backend;
    # set these to any placeholder — they are NOT used here.
    POSTGRES_USER=_unused_
    POSTGRES_PASSWORD=_unused_

Optional:
    RUN_CLEANUP=false   # set to true to delete test data on next run
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from quanta.config import get_settings
from quanta import (
    MultiRetriever,
    Neo4jGraph,
    QuantaIndex,
    RetrievalResult,
    get_docstore,
)
from quanta.docstore import DocStoreBackend

# ── Document catalogue ─────────────────────────────────────────────────────────

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc_001",
        "title": "Urban Land Use and Zoning Act",
        "doc_type": "law",
        "content": (
            "The Urban Land Use and Zoning Act establishes the foundational legal framework governing the "
            "allocation and regulation of land within metropolitan areas, defining permissible uses for "
            "residential, commercial, industrial, mixed-use, and open-space zones and mandating that "
            "municipal authorities develop comprehensive zoning maps subject to review every ten years. "
            "The Act introduces floor-area ratio limits, setback requirements, and building height "
            "restrictions to preserve neighbourhood character while enabling density targets aligned with "
            "regional housing demand projections. Any rezoning proposal exceeding five hectares must "
            "undergo a full environmental impact assessment and a minimum of two public consultation "
            "rounds before the city council may approve it. Amendments to the zoning map must be "
            "published in the official municipal register and made accessible through the city's open "
            "data portal within thirty days of adoption. The Act further provides a variance and "
            "special-use permit process allowing deviations from standard zoning rules subject to "
            "demonstrated public benefit, neighbour notification periods of no fewer than twenty-one "
            "days, and independent planning board review, ensuring that exceptional cases are handled "
            "transparently and consistently without undermining the integrity of the base zoning code."
        ),
    },
    {
        "id": "doc_002",
        "title": "Sustainable Urban Mobility Plan (SUMP)",
        "doc_type": "plan",
        "content": (
            "The Sustainable Urban Mobility Plan sets out a ten-year strategy for transforming the city's "
            "transport system toward lower emissions, greater safety, and improved access for all "
            "residents, with measurable mode-shift targets requiring that sixty percent of all daily "
            "trips be made by walking, cycling, or public transit by 2033. The plan prioritises expansion "
            "of the public transit network through new metro lines, bus rapid transit corridors, and "
            "on-demand microtransit services in low-density peripheries underserved by fixed-route "
            "operations. Cycling infrastructure is targeted at a two-hundred-kilometre dedicated lane "
            "network complemented by a city-wide bike-share programme with stations at every major "
            "transit interchange and employment centre. A congestion-pricing scheme in the central "
            "business district will generate revenues ringfenced for transit subsidies, pedestrian realm "
            "improvements, and accessibility upgrades at existing stops. The plan mandates mobility "
            "impact assessments for any development generating more than five hundred daily vehicle trips "
            "and requires coordination with the land-use authority to ensure transit-oriented development "
            "around station precincts. Equity provisions guarantee discounted transit passes for "
            "low-income residents and require accessibility audits at all new and refurbished stops to "
            "ensure compliance with universal design standards."
        ),
    },
    {
        "id": "doc_003",
        "title": "Affordable Housing Policy Framework",
        "doc_type": "policy",
        "content": (
            "The Affordable Housing Policy Framework establishes the city's commitment to ensuring that "
            "at least twenty percent of all new residential developments above fifty units include units "
            "priced at or below eighty percent of area median income, with deeper affordability tiers "
            "required in high-opportunity zones identified by the planning authority. The framework "
            "introduces an inclusionary zoning mandate and creates a Housing Trust Fund capitalised "
            "through developer in-lieu fees, public land contributions, and tax-increment financing "
            "revenues from designated growth corridors. Affordable Housing Overlay Zones in "
            "high-opportunity neighbourhoods with good transit access offer density bonuses of up to "
            "forty percent in exchange for affordability commitments binding on subsequent owners for a "
            "minimum of fifty years. Anti-displacement measures protect existing tenants through "
            "right-to-return policies, community land trust partnerships, and a tenant assistance fund "
            "providing legal advice and emergency rental support. An annual progress report tracks units "
            "delivered, income levels served, geographic distribution, and displacement rates, reported "
            "to the city council and published on the open data portal for independent analysis."
        ),
    },
    {
        "id": "doc_004",
        "title": "Smart City Strategic Plan 2030",
        "doc_type": "strategy",
        "content": (
            "The Smart City Strategic Plan 2030 outlines the city's vision for leveraging digital "
            "technologies to improve service delivery, environmental sustainability, and civic "
            "participation across all municipal functions. Core initiatives include deployment of a "
            "city-wide Internet of Things sensor network for real-time monitoring of air quality, "
            "traffic flows, noise levels, waste fill levels, and energy consumption in public buildings, "
            "with all sensor data aggregated on a unified urban data platform accessible to city "
            "departments and, in anonymised form, to the public. A digital twin of the urban fabric "
            "allows planners to simulate the impact of infrastructure investments, zoning changes, and "
            "climate events before committing public capital to construction. Governance provisions "
            "establish a Data Ethics Board to oversee data collection protocols, privacy compliance, "
            "algorithmic decision-making in public services, and the city's obligations under data "
            "protection legislation. The strategy funds digital-literacy programmes in underserved "
            "communities to ensure equitable access to smart city benefits, and establishes an "
            "innovation procurement framework enabling start-ups and civic-tech organisations to pilot "
            "solutions in a regulatory sandbox environment, with successful pilots scaled city-wide."
        ),
    },
    {
        "id": "doc_005",
        "title": "Urban Climate Resilience Framework",
        "doc_type": "framework",
        "content": (
            "The Urban Climate Resilience Framework provides a comprehensive assessment of the city's "
            "exposure to climate-related hazards—including extreme heat events, flash flooding, coastal "
            "storm surge, prolonged drought, and compound weather events—and establishes adaptation "
            "targets and investment priorities through 2050. All new public infrastructure must be "
            "designed to withstand a one-in-one-hundred-year storm event, while critical facilities such "
            "as hospitals, emergency shelters, water treatment plants, and energy substations must meet "
            "enhanced resilience standards equivalent to a one-in-two-hundred-year event. The framework "
            "introduces a Green-Blue Infrastructure Fund to co-finance permeable pavements, bioswales, "
            "retention ponds, urban tree canopy expansion, and coastal wetland restoration, with "
            "allocations weighted toward the most climate-vulnerable neighbourhoods as identified by "
            "the city's spatial risk mapping tool. Building codes are updated to require cool roofs, "
            "passive cooling features, and flood-resilient construction in all new buildings above two "
            "storeys. Community resilience hubs established in the twenty most vulnerable neighbourhoods "
            "provide emergency cooling and warming centres, backup power generation, clean water reserves, "
            "and local food storage to sustain residents through extended disruption events."
        ),
    },
    {
        "id": "doc_006",
        "title": "Public Participation in Urban Planning Guidelines",
        "doc_type": "guideline",
        "content": (
            "The Public Participation in Urban Planning Guidelines define minimum standards for community "
            "engagement throughout the planning and policy-making cycle, applying to all plans, policies, "
            "and major development decisions made by the municipal planning authority. Major plans "
            "affecting areas over two hectares or five hundred residents must undergo at least three "
            "structured engagement phases: early scoping workshops to identify community values and "
            "priorities, a public exhibition of draft proposals with a minimum four-week comment period, "
            "and a formal hearing before the planning board prior to adoption. Key documents must be "
            "translated into the four most widely spoken languages in the city, and dedicated outreach "
            "is required for renters, youth, elderly residents, people with disabilities, and ethnic "
            "minority communities who are historically underrepresented in formal planning processes. "
            "Online engagement platforms must meet WCAG 2.1 AA accessibility standards and be "
            "supplemented by in-person events held in community venues within each affected neighbourhood "
            "at times accessible to working residents. Planners must publish a Statement of Community "
            "Involvement within sixty days of plan adoption, documenting how feedback shaped the final "
            "decision and transparently explaining cases where community preferences were not followed."
        ),
    },
    {
        "id": "doc_007",
        "title": "Zoning Reform Commission Report",
        "doc_type": "report",
        "content": (
            "The Zoning Reform Commission Report presents findings from an eighteen-month independent "
            "review of the city's existing zoning code, identifying regulatory barriers to housing "
            "production, mixed-use development, climate-adaptive construction, and active transport "
            "infrastructure. The Commission concludes that single-family-only zones covering sixty "
            "percent of the residential land base are inconsistent with regional housing targets and "
            "recommends their abolition in favour of a gentle-density baseline permitting up to four "
            "dwellings per lot by right, removing the need for discretionary approval in most cases. "
            "Mandatory minimum parking requirements are recommended for elimination citywide, supported "
            "by evidence that parking minimums increase construction costs by up to twenty percent, "
            "reduce housing supply, undermine walkability, and directly conflict with the Sustainable "
            "Urban Mobility Plan's mode-shift goals. Form-based coding standards are proposed for all "
            "priority growth corridors identified in the regional plan, replacing prescriptive use tables "
            "with design-oriented standards for massing, frontage, and ground-floor activation. A "
            "streamlined ministerial approval pathway for fully affordable housing projects is expected "
            "to reduce approval timelines from an average of thirty-six months to under twelve, "
            "significantly accelerating delivery against the Affordable Housing Policy Framework targets."
        ),
    },
    {
        "id": "doc_008",
        "title": "Green Infrastructure and Urban Ecology Standards",
        "doc_type": "standard",
        "content": (
            "The Green Infrastructure and Urban Ecology Standards establish mandatory minimum requirements "
            "for integrating nature-based solutions into all new urban development and public realm "
            "projects funded, approved, or delivered by the municipality. Developments exceeding one "
            "thousand square metres of gross floor area must provide on-site permeable surfaces covering "
            "at least thirty percent of the site area, achieve a Biodiversity Net Gain score of at least "
            "ten percent above the pre-development baseline, and plant a minimum of one tree per two "
            "hundred square metres of gross floor area using species from the approved climate-adapted "
            "planting palette. Street design standards require a minimum four-metre-wide planting strip "
            "along all arterial roads, with species selected for heat tolerance, air quality improvement, "
            "stormwater absorption, and native wildlife habitat provision. A Canopy Cover Target commits "
            "the city to achieving forty percent tree canopy coverage by 2040, monitored through annual "
            "satellite imagery analysis and reported publicly. Developers disturbing trees above a "
            "specified diameter must provide replacement planting at a three-to-one ratio within the "
            "same neighbourhood and pay into a canopy restoration fund when on-site replacement is "
            "not feasible due to site constraints."
        ),
    },
    {
        "id": "doc_009",
        "title": "Transit-Oriented Development Policy",
        "doc_type": "policy",
        "content": (
            "The Transit-Oriented Development Policy directs that all land within eight hundred metres "
            "of major transit stations be planned and regulated to achieve minimum residential densities "
            "of eighty dwellings per hectare and minimum employment densities of three thousand square "
            "metres of commercial floor space per hectare, creating walkable, mixed-use precincts that "
            "maximise ridership and reduce car dependence at the network level. Active ground-floor uses "
            "including retail, food and beverage, childcare, health services, and community facilities "
            "are mandated along all station-facing frontages to generate continuous pedestrian activity "
            "and provide essential services within walking distance of transit for daily needs. Joint "
            "planning committees comprising transit operators, the planning department, council, and "
            "affected landowners must prepare precinct structure plans for each major station catchment "
            "within three years of station opening or rezoning. Height and design controls preserve "
            "view corridors, solar access for adjacent residential buildings, and wind comfort at street "
            "level through wind tunnel testing requirements for towers above twelve storeys. Developer "
            "contributions within TOD precincts are directed exclusively to pedestrian and cycling "
            "infrastructure, public plazas, and affordable housing consistent with the Affordable "
            "Housing Policy Framework targets."
        ),
    },
    {
        "id": "doc_010",
        "title": "Urban Heat Island Mitigation Strategy",
        "doc_type": "strategy",
        "content": (
            "The Urban Heat Island Mitigation Strategy responds to evidence that the city's densest "
            "built-up areas experience average temperatures two to four degrees Celsius above the "
            "surrounding rural fringe during heatwave events, resulting in excess mortality among "
            "elderly and vulnerable residents, increased peak energy demand, higher cooling costs, and "
            "significantly reduced outdoor liveability and productivity. The strategy mandates cool "
            "roofs and cool pavements in all new developments and major refurbishments, prescribing "
            "minimum solar reflectance index values in updated building regulations with a compliance "
            "verification process at the point of development approval. An urban greening corridors "
            "programme creates shaded pedestrian and cycling paths linking parks, street trees, green "
            "roofs, and wetlands, targeting a maximum two-hundred-metre distance between any resident "
            "and a shaded active-travel route within ten years. A Heat Vulnerability Index mapping "
            "tool identifies priority neighbourhoods for immediate greening investment and emergency "
            "cooling infrastructure deployment, weighting areas with elderly populations, low tree "
            "canopy, high building density, and limited green space access. Implementation is "
            "coordinated with the Urban Climate Resilience Framework, the Green Infrastructure and "
            "Urban Ecology Standards, and the Sustainable Urban Mobility Plan's active travel goals."
        ),
    },
    {
        "id": "doc_011",
        "title": "Inclusive Urban Regeneration Programme",
        "doc_type": "programme",
        "content": (
            "The Inclusive Urban Regeneration Programme targets ten priority neighbourhoods characterised "
            "by concentrations of derelict buildings, elevated unemployment, poor health outcomes, "
            "inadequate public amenities, and high levels of social deprivation as measured by the "
            "city's composite neighbourhood index. The programme deploys a Place-Based Investment model "
            "combining physical improvements—streetscape upgrades, community centre construction, park "
            "creation, and active frontage activation—with social services co-located in new facilities "
            "including employment support, health clinics, mental health services, and subsidised "
            "childcare. Anti-displacement clauses guarantee existing residents and businesses the right "
            "of return after regeneration works, with temporary relocation assistance funded from the "
            "programme budget at rates covering comparable local accommodation. Community Land Trusts "
            "are supported as vehicles for permanently affordable housing and commercial space, removing "
            "assets from speculative land markets and locking in affordability in perpetuity. Every "
            "major development project in target areas must sign a Community Benefit Agreement "
            "negotiated with resident-led neighbourhood boards, covering local hiring targets, "
            "apprenticeship programmes, long-term maintenance commitments, and contributions to "
            "community facilities."
        ),
    },
    {
        "id": "doc_012",
        "title": "Digital Urban Infrastructure Act",
        "doc_type": "law",
        "content": (
            "The Digital Urban Infrastructure Act creates a comprehensive regulatory framework for the "
            "installation, operation, and governance of digital infrastructure in public spaces, covering "
            "5G small cells, smart streetlights, environmental and mobility sensors, automated traffic "
            "management systems, public Wi-Fi networks, and any future connected device categories "
            "deployed in the public right of way. All infrastructure providers must obtain a Digital "
            "Street Works permit and demonstrate compliance with electromagnetic field safety standards, "
            "aesthetic integration guidelines, underground utility coordination requirements, and data "
            "governance obligations before any installation commences. A mandatory data-sharing protocol "
            "ensures that data from city-funded digital infrastructure is accessible to the municipal "
            "government under a public interest licence, enabling use in the urban data platform and "
            "digital twin without additional cost. An independent Digital Infrastructure Regulator is "
            "established to handle licensing, compliance monitoring, dispute adjudication, and annual "
            "reporting on network coverage and data quality. Cybersecurity provisions require all "
            "connected infrastructure to meet national critical-infrastructure protection standards, "
            "undergo annual penetration testing, and report security incidents to the Regulator within "
            "twenty-four hours of detection."
        ),
    },
    {
        "id": "doc_013",
        "title": "Municipal Economic Development Zone Regulations",
        "doc_type": "regulation",
        "content": (
            "The Municipal Economic Development Zone Regulations establish the criteria, governance "
            "framework, and performance accountability mechanisms for designating, managing, and "
            "evaluating Special Economic Development Zones within the city's administrative boundary. "
            "Zones are designated by the city council on the recommendation of the Economic Development "
            "Board to attract investment in target sectors including advanced manufacturing, clean "
            "technology, creative industries, professional services, and logistics, with each zone "
            "required to articulate a sector specialisation strategy aligned with the city's long-term "
            "economic diversification goals. Designated businesses receive a ten-year rates relief "
            "package, expedited planning approvals through a dedicated zone planning officer, and access "
            "to a shared industrial infrastructure fund for roads, energy connections, and digital "
            "infrastructure upgrades. Zone management bodies must publish annual economic impact reports "
            "covering jobs created, average wages paid, local procurement rates, carbon emissions, and "
            "female and minority business participation. Sunset provisions require each zone to "
            "demonstrate net positive fiscal impact within seven years or face designation review. "
            "All zone development must meet Smart City infrastructure readiness requirements and the "
            "energy and flood resilience standards set out in the Urban Climate Resilience Framework."
        ),
    },
    {
        "id": "doc_014",
        "title": "Waterfront Revitalisation Master Plan",
        "doc_type": "plan",
        "content": (
            "The Waterfront Revitalisation Master Plan guides the transformation of fourteen kilometres "
            "of underused industrial and port land into a mixed-use, publicly accessible waterfront "
            "precinct over a twenty-year horizon, creating a new city quarter that balances economic "
            "activation, public amenity, ecological restoration, and climate resilience. The plan "
            "establishes a continuous public foreshore promenade with legally binding view corridors to "
            "the water, active recreation areas, cultural facilities including a new maritime museum and "
            "outdoor performance spaces, and a marina precinct for recreational boating. Thirty percent "
            "of all residential development in the precinct must be affordable housing delivered through "
            "the mechanisms of the Affordable Housing Policy Framework, ensuring the new quarter is "
            "socially mixed from the outset. Flood resilience is addressed through a foreshore seawall "
            "upgrade designed to the one-in-five-hundred-year standard and a series of tidal wetland "
            "buffers that provide ecological habitat, carbon sequestration, and passive amenity for "
            "residents. Heritage-listed industrial structures are designated for adaptive reuse, with "
            "demolition prohibited without a demonstrated conservation plan approved by the Heritage "
            "Advisory Panel. An integrated mobility hub at the waterfront's central node provides ferry, "
            "bus, cycling, and pedestrian connections consistent with the Sustainable Urban Mobility Plan."
        ),
    },
    {
        "id": "doc_015",
        "title": "Urban Air Quality Management Policy",
        "doc_type": "policy",
        "content": (
            "The Urban Air Quality Management Policy establishes ambient concentration targets for fine "
            "particulate matter (PM2.5), nitrogen dioxide (NO2), and ground-level ozone that are fully "
            "aligned with World Health Organisation 2021 guidelines and more stringent than the current "
            "national minimum standards, reflecting the city council's determination to protect public "
            "health beyond the regulatory floor. A Low Emission Zone covering the inner city prohibits "
            "diesel vehicles older than Euro 6 standards from operating within the zone boundary during "
            "peak hours, with a 2035 target date for extending the zone to all non-zero-emission private "
            "vehicles. Industrial and construction site emission permits are tightened with mandatory "
            "continuous monitoring and public reporting requirements, and a real-time air quality "
            "monitoring network linked to the Smart City IoT sensor platform publishes hourly data and "
            "triggers tiered public health advisories when thresholds are exceeded. A Clean Air Fund "
            "supports retrofitting of heavy goods vehicles operated by small businesses, subsidises "
            "transition to electric fleet vehicles for city service contractors, and funds air quality "
            "improvement measures in schools and health facilities in the most polluted neighbourhoods. "
            "The policy requires annual reporting to the city council on attainment of WHO guideline "
            "values and a published roadmap to full compliance by 2035."
        ),
    },
    {
        "id": "doc_016",
        "title": "Equity and Social Justice in Urban Planning Report",
        "doc_type": "report",
        "content": (
            "The Equity and Social Justice in Urban Planning Report presents findings from a city-wide "
            "audit of planning decisions over the past two decades, examining whether infrastructure "
            "investment, green space provision, zoning designations, and major development decisions "
            "have systematically disadvantaged low-income residents, ethnic minorities, recent migrants, "
            "and other marginalised groups. The audit finds significant and persistent spatial "
            "inequalities in park access, school quality proximity, public transport coverage, broadband "
            "connectivity, and exposure to industrial pollution and urban heat, concentrated in a "
            "consistent set of twenty under-served neighbourhoods. The report recommends mandatory "
            "Equity Impact Assessments for all major planning decisions, an Equitable Infrastructure "
            "Investment Framework directing at least forty percent of capital expenditure to the twenty "
            "percent most deprived neighbourhoods, and establishment of a permanent Resident Advisory "
            "Panel with meaningful decision-making authority over neighbourhood-level plans. It further "
            "recommends revisions to the Public Participation Guidelines to introduce proportional "
            "representation of marginalised communities on all statutory planning advisory bodies and "
            "to fund community capacity-building so that residents can engage as informed participants "
            "rather than passive consultees in shaping the plans that govern their neighbourhoods."
        ),
    },
    {
        "id": "doc_017",
        "title": "Historic District Preservation Guidelines",
        "doc_type": "guideline",
        "content": (
            "The Historic District Preservation Guidelines set out standards for development, alteration, "
            "and maintenance of buildings and public realm within the city's thirty-two designated "
            "Heritage Conservation Areas, which together contain over twelve thousand listed buildings "
            "and structures of architectural, historical, and cultural significance. New construction "
            "within Heritage Conservation Areas must be compatible in scale, massing, materials, and "
            "architectural character with the prevailing historic context while avoiding direct "
            "replication of historic styles in a pastiche manner that misrepresents the area's authentic "
            "evolution. Alterations to individually listed buildings must use reversible methods wherever "
            "technically feasible, retain all original fabric of significance, and receive prior written "
            "approval from the Heritage Advisory Panel, which must determine applications within sixty "
            "days. A Conservation Area Character Appraisal process is undertaken every ten years to "
            "identify significance, vulnerability, pressures, and enhancement opportunities, with "
            "findings informing updates to the area's management plan. Tax incentives and heritage "
            "repair grants are available to owners of listed buildings undertaking approved conservation "
            "works using traditional materials and craftsmanship. The guidelines address integration of "
            "solar panels, heat pumps, and smart building technologies into historic structures, "
            "permitting them where installations are reversible and do not harm significant fabric."
        ),
    },
    {
        "id": "doc_018",
        "title": "Urban Food Security and Community Agriculture Policy",
        "doc_type": "policy",
        "content": (
            "The Urban Food Security and Community Agriculture Policy addresses the acute challenge of "
            "food deserts in the city's outer suburbs and deprived inner-city neighbourhoods, where "
            "residents lack access to fresh, affordable, and nutritious food within walkable distance "
            "and are disproportionately dependent on energy-dense convenience food from petrol stations "
            "and fast-food outlets. The policy commits to establishing at least one community garden, "
            "urban farm, or food hub within every five-hundred-metre catchment of a food-insecure census "
            "area within five years, with progress tracked through an annual food environment mapping "
            "exercise published on the open data portal. Planning approvals for food-producing land uses "
            "on vacant publicly owned land are streamlined to a delegated officer decision within eight "
            "weeks, and rooftop agriculture on commercial buildings is permitted subject to structural "
            "assessment and basic safety standards. A Community Agriculture Support Programme provides "
            "grants, training, soil testing, and technical assistance to neighbourhood-run food growing "
            "initiatives. The policy explicitly aligns with the Urban Climate Resilience Framework by "
            "recognising urban agriculture's co-benefits for heat mitigation, stormwater absorption, "
            "and biodiversity, and designates the Inclusive Urban Regeneration Programme as the primary "
            "delivery vehicle for food hubs in the ten priority neighbourhoods."
        ),
    },
    {
        "id": "doc_019",
        "title": "Urban Noise Pollution Control Ordinance",
        "doc_type": "regulation",
        "content": (
            "The Urban Noise Pollution Control Ordinance establishes legally binding ambient noise limits "
            "for residential, mixed-use, and commercial zones differentiated by time of day, setting "
            "daytime limits of fifty-five decibels and night-time limits of forty-five decibels at the "
            "façade of sensitive receptors including dwellings, schools, hospitals, and care homes. "
            "All major development proposals within two hundred metres of identified noise sources—"
            "including arterial roads, rail lines, entertainment districts, and industrial operations—"
            "must submit an acoustic impact assessment demonstrating compliance through a combination "
            "of source reduction, barrier design, and building fabric treatment. A city-wide smart "
            "noise monitoring network, integrated with the Smart City IoT platform, continuously "
            "measures noise levels at strategic points and feeds data to the environmental health "
            "department for enforcement prioritisation and public information. The Ordinance introduces "
            "a Quiet Streets designation programme under which low-traffic residential streets meeting "
            "defined acoustic and active-travel criteria may be upgraded with traffic calming, planting, "
            "and seating to create peaceful public spaces. Provisions on night-time economy operations "
            "require noise management plans, sound insulation standards for new licensed premises, and "
            "curfew limits on amplified music and outdoor events, with repeat violators subject to "
            "licence review and graduated financial penalties."
        ),
    },
    {
        "id": "doc_020",
        "title": "Circular Economy and Municipal Waste Strategy",
        "doc_type": "strategy",
        "content": (
            "The Circular Economy and Municipal Waste Strategy commits the city to achieving a "
            "seventy-five percent household waste recycling and composting rate by 2030 and to "
            "eliminating all avoidable single-use plastics from city-funded events, facilities, and "
            "procurement by 2027, as part of a wider transition from the linear take-make-dispose model "
            "to a regenerative urban economy that retains the value of materials, components, and "
            "products for as long as possible. The strategy introduces mandatory food waste separation "
            "and kerbside collection for all households and businesses above a threshold size, with "
            "collected organics processed through anaerobic digestion to produce biogas and digestate "
            "for urban agriculture. A Repair and Reuse Hub network provides residents with spaces and "
            "tools for fixing household items, swapping goods, and learning repair skills, supported "
            "by social enterprise operators. Extended producer responsibility provisions leverage the "
            "city's procurement power to incentivise manufacturers operating in the city to adopt "
            "take-back schemes and recyclability-by-design standards. Smart waste monitoring sensors "
            "on public litter bins and collection vehicles, integrated with the Smart City data "
            "platform, optimise collection routes and identify illegal dumping hotspots in real time, "
            "reducing fleet kilometres and associated carbon emissions while improving service quality."
        ),
    },
    {
        "id": "doc_021",
        "title": "Integrated Urban Water Management Policy",
        "doc_type": "policy",
        "content": (
            "The Integrated Urban Water Management Policy establishes a whole-of-water-cycle approach "
            "to planning and managing the city's water resources, stormwater systems, and wastewater "
            "infrastructure in the face of intensifying climate variability, ageing pipe networks, and "
            "growing demand from a rising urban population. The policy requires all new developments "
            "of more than five hundred square metres to incorporate on-site stormwater detention "
            "achieving a peak flow reduction of at least fifty percent compared to pre-development "
            "runoff, using a combination of permeable surfaces, rainwater harvesting, rain gardens, and "
            "underground storage tanks to reduce pressure on combined sewer overflows. A Water Sensitive "
            "Urban Design standard is adopted for all public realm projects, requiring blue-green "
            "infrastructure elements such as bioswales, constructed wetlands, and tree pits with "
            "structural cells to manage rainfall at source rather than routing it directly to grey "
            "infrastructure. Mandatory water efficiency standards for new buildings require dual-pipe "
            "systems for non-potable uses and rainwater harvesting for irrigation and toilet flushing. "
            "A city-wide asset management programme for water and sewer networks prioritises renewal of "
            "the highest-risk pipe segments based on hydraulic modelling and condition surveys, reducing "
            "the frequency of sewer flooding events and protecting watercourses from combined sewer "
            "overflow pollution."
        ),
    },
    {
        "id": "doc_022",
        "title": "Walkable Neighbourhoods and Active Streets Programme",
        "doc_type": "programme",
        "content": (
            "The Walkable Neighbourhoods and Active Streets Programme reorients the city's street "
            "design philosophy from vehicle throughput optimisation to the creation of safe, comfortable, "
            "and attractive environments for pedestrians and cyclists of all ages and abilities, guided "
            "by the principle that every resident should be able to meet their daily needs—shopping, "
            "education, healthcare, recreation, and transit access—within a fifteen-minute walk from "
            "home. The programme introduces a hierarchical street typology with detailed design standards "
            "for each type, from high-footfall civic boulevards with generous footways and tree canopy "
            "to low-traffic residential streets with filtered permeability and play streets. A Pedestrian "
            "Priority Zone is established in the city centre, progressively reducing motor vehicle "
            "access during peak pedestrian periods while maintaining servicing and emergency access. "
            "School streets closures during drop-off and pick-up times reduce child exposure to vehicle "
            "exhaust and road injury risk and are supported by an active travel education programme in "
            "all primary schools. Capital investment is prioritised by a walkability gap analysis "
            "identifying footway deficiencies, missing crossing points, and poor lighting in areas with "
            "high pedestrian demand but poor current infrastructure, with equity weighting directing "
            "funding toward the most deprived neighbourhoods first."
        ),
    },
    {
        "id": "doc_023",
        "title": "Urban Biodiversity and Ecological Connectivity Plan",
        "doc_type": "plan",
        "content": (
            "The Urban Biodiversity and Ecological Connectivity Plan maps the city's existing ecological "
            "assets—parks, street trees, gardens, watercourses, green roofs, and informal green spaces—"
            "and establishes a strategic network of ecological corridors linking these assets to allow "
            "movement of wildlife, particularly pollinators, birds, and small mammals, through an "
            "otherwise heavily fragmented urban landscape. The plan designates fifteen priority "
            "ecological corridors where development must incorporate wildlife-friendly design features "
            "including hedgehog highways, bat boxes, swift bricks, native planting, and reduced lighting "
            "levels to protect nocturnal species from light pollution impacts. A city-wide citizen "
            "science programme engages residents in recording species sightings through a dedicated "
            "urban nature app, building a live biodiversity database that informs planning decisions "
            "and tracks progress against habitat connectivity targets over time. All major development "
            "projects within or adjacent to ecological corridors must complete a biodiversity impact "
            "assessment and achieve measurable net gain through on-site and off-site habitat creation "
            "or restoration, verified by an independent ecologist at both pre-commencement and "
            "post-completion stages. The plan aligns with the Green Infrastructure and Urban Ecology "
            "Standards and the Urban Climate Resilience Framework, recognising biodiversity as "
            "integral to the city's long-term ecological function and climate adaptation capacity."
        ),
    },
    {
        "id": "doc_024",
        "title": "Night-Time Economy and Entertainment District Policy",
        "doc_type": "policy",
        "content": (
            "The Night-Time Economy and Entertainment District Policy recognises the city's after-dark "
            "cultural and hospitality sector as a significant contributor to economic output, employment, "
            "cultural identity, and social wellbeing, while establishing a governance framework that "
            "actively manages the negative impacts of concentrated late-night activity including noise, "
            "litter, anti-social behaviour, and adverse effects on resident amenity in mixed-use areas. "
            "Four designated Entertainment Districts are established where evening and late-night uses "
            "are supported and managed through a District Partnership model that brings together "
            "licensees, residents, cultural organisations, transit operators, and the police under a "
            "shared management plan updated every three years. A Night Mayor is appointed as an "
            "independent advocate for the night-time economy, with a mandate to mediate conflicts, "
            "advise licensing committees, promote diverse and inclusive after-dark culture, and monitor "
            "the health and safety of the after-dark public realm. Planning policies require sound "
            "insulation standards in new mixed-use buildings, mandatory noise impact assessments for "
            "new licensed premises above a threshold capacity, and a Circular Economy-aligned waste "
            "management plan for every venue. Safe transport connections including extended public "
            "transit services and regulated taxi ranks are coordinated with the Sustainable Urban "
            "Mobility Plan's late-night service provisions."
        ),
    },
    {
        "id": "doc_025",
        "title": "Municipal Energy Transition and Renewable Energy Plan",
        "doc_type": "plan",
        "content": (
            "The Municipal Energy Transition and Renewable Energy Plan sets out the pathway to "
            "decarbonising the city's energy system by 2045, covering municipal buildings and "
            "operations, the district heating and cooling network, street lighting, public transport "
            "electrification, and the frameworks enabling private sector and household energy "
            "transition at scale. All new municipal buildings must achieve net-zero operational carbon "
            "from the date of first occupation, using a fabric-first approach, heat pump technology, "
            "rooftop solar, and battery storage to minimise energy demand before meeting remaining "
            "needs with on-site or locally sourced renewables. A city-wide district heating expansion "
            "programme connects dense residential and commercial areas to low-carbon heat networks "
            "powered by waste heat recovery, large-scale heat pumps drawing on the river as a heat "
            "source, and geothermal energy where geological conditions permit. A Solar City programme "
            "supports rooftop solar installation on private homes and businesses through a community "
            "bulk-purchasing scheme, a low-interest green loan facility, and simplified grid connection "
            "permitting. The plan explicitly references the Municipal Net-Zero Carbon Roadmap as the "
            "overarching strategic framework and requires all city-owned energy assets to report "
            "annually against decarbonisation milestones using standardised carbon accounting methods."
        ),
    },
    {
        "id": "doc_026",
        "title": "Homelessness Prevention and Housing First Strategy",
        "doc_type": "strategy",
        "content": (
            "The Homelessness Prevention and Housing First Strategy reframes the city's response to "
            "homelessness from crisis management toward systematic prevention and rapid, permanent "
            "rehousing, guided by the internationally evidenced Housing First principle that stable "
            "housing is a prerequisite for addressing the complex needs associated with long-term "
            "homelessness, rather than a reward for achieving other outcomes. The strategy commits to "
            "ending rough sleeping within five years through a combination of assertive outreach teams "
            "with sustained caseloads, a twenty-four-hour crisis response service, and a guaranteed "
            "Housing First offer for every person sleeping rough on any given night. Prevention "
            "services intervene at key risk points including eviction proceedings, hospital discharge, "
            "prison release, and care leavers turning eighteen, providing emergency financial assistance, "
            "tenancy mediation, and rapid rehousing brokerage before the loss of accommodation occurs. "
            "A pipeline of supported housing and move-on accommodation is created in partnership with "
            "registered housing providers, drawing on affordable housing contributions from the "
            "Affordable Housing Policy Framework. The strategy requires integration of mental health, "
            "substance use, employment, and immigration support services within Housing First teams, "
            "and commits to annual transparent reporting on rough sleeping counts, rehousing outcomes, "
            "and tenancy sustainment rates, with an independent evaluation at three years."
        ),
    },
    {
        "id": "doc_027",
        "title": "Child and Youth Inclusive Urban Design Guidelines",
        "doc_type": "guideline",
        "content": (
            "The Child and Youth Inclusive Urban Design Guidelines establish standards and principles "
            "for planning and designing urban environments that prioritise the safety, independence, "
            "playfulness, and social development of children and young people, recognising that cities "
            "designed for the most vulnerable users become better cities for everyone. The guidelines "
            "require that all new public open spaces above five hundred square metres incorporate "
            "designated play provision appropriate for multiple age groups—from toddler areas with "
            "natural play materials to youth spaces with equipment and social settings for teenagers—"
            "designed with input from children and young people through structured participatory design "
            "processes. Streets adjacent to schools, parks, and community centres must meet the Safe "
            "Routes to School standard, including traffic-calmed approaches, raised crossings, and "
            "continuous footways with no kerb-to-carriageway conflicts. Planning applications for new "
            "residential developments above fifty units must submit a Child Impact Assessment evaluating "
            "access to play space, schools, community facilities, and safe walking and cycling routes, "
            "with shortfalls addressed through developer contributions. The guidelines align with the "
            "Walkable Neighbourhoods and Active Streets Programme and the Equity and Social Justice in "
            "Urban Planning Report's findings on disproportionate play space deficits in deprived areas."
        ),
    },
    {
        "id": "doc_028",
        "title": "Age-Friendly City Action Plan",
        "doc_type": "plan",
        "content": (
            "The Age-Friendly City Action Plan adopts the World Health Organisation's Age-Friendly "
            "Cities framework and translates it into a five-year programme of actions across eight "
            "domains—outdoor spaces and buildings, transportation, housing, social participation, "
            "respect and social inclusion, civic participation and employment, communication and "
            "information, and community support and health services—with the explicit goal of enabling "
            "older residents to live healthy, active, and socially connected lives in their own homes "
            "and neighbourhoods for as long as they choose. The plan requires that all new public "
            "buildings, public realm projects, and transport infrastructure meet enhanced accessibility "
            "standards including step-free access, adequate rest seating at no greater than fifty-metre "
            "intervals on principal walking routes, legible signage, and lighting meeting the "
            "standards for low-vision users. A home adaptation grant scheme co-funded by the health "
            "authority supports the installation of grab rails, stair lifts, wet rooms, and "
            "assistive technology for older homeowners and private renters, reducing hospital admissions "
            "from falls and enabling earlier discharge from acute care. Community social prescribing "
            "services connect isolated older residents to local voluntary activities, exercise "
            "programmes, and befriending schemes, addressing the public health consequences of "
            "loneliness. The plan was co-designed with the Older Persons Advisory Group and requires "
            "annual reporting to the city council with outcome data disaggregated by age, gender, and "
            "neighbourhood deprivation level."
        ),
    },
    {
        "id": "doc_029",
        "title": "Brownfield Regeneration and Industrial Land Repurposing Strategy",
        "doc_type": "strategy",
        "content": (
            "The Brownfield Regeneration and Industrial Land Repurposing Strategy identifies over "
            "eight hundred hectares of vacant, underused, or contaminated former industrial land "
            "across the city as the priority location for new housing, employment, and mixed-use "
            "development over the next twenty years, directing growth inward to protect the green "
            "belt and reduce infrastructure costs associated with peripheral greenfield development. "
            "The strategy establishes a Brownfield Land Register maintained by the planning authority "
            "and updated annually, classifying sites by suitability for immediate development, "
            "development subject to remediation, or long-term strategic reserve, enabling proactive "
            "engagement with landowners and targeted public investment in enabling infrastructure. "
            "A Contaminated Land Remediation Fund provides grant and loan finance for the investigation "
            "and clean-up of severely contaminated sites where remediation costs would otherwise make "
            "development unviable, unlocking land for housing and employment uses while eliminating "
            "risks to groundwater and public health. Planning policies for brownfield sites require "
            "phased delivery strategies, infrastructure-first sequencing, and design that responds to "
            "the industrial heritage of each site. The strategy aligns with the Affordable Housing "
            "Policy Framework's requirement for twenty percent affordable provision on all sites above "
            "fifty units, the Urban Climate Resilience Framework's flood risk and energy standards, "
            "and the Urban Air Quality Management Policy's requirements for remediation of historically "
            "polluted soils before residential development commences."
        ),
    },
    {
        "id": "doc_030",
        "title": "Municipal Net-Zero Carbon Roadmap",
        "doc_type": "strategy",
        "content": (
            "The Municipal Net-Zero Carbon Roadmap establishes the city's legally binding commitment "
            "to reach net-zero greenhouse gas emissions across all municipal operations and, through "
            "enabling policies and partnership programmes, to support the wider city economy and "
            "community to reach net-zero by 2045, consistent with a 1.5-degree warming pathway. "
            "The roadmap quantifies the emissions reduction potential of each major intervention "
            "sector—buildings, transport, energy supply, waste, land use, and consumption—and sets "
            "five-year carbon budgets for the city with independent annual verification by a Climate "
            "Advisory Panel reporting publicly to the city council. Priority actions for the first "
            "budget period include deep retrofit of the worst-performing social housing stock to EPC "
            "band B, full electrification of the municipal vehicle fleet, completion of the solar "
            "city programme, and a carbon sequestration programme through urban forest expansion and "
            "wetland restoration. A Carbon Literacy Programme delivers training to all council staff "
            "and elected members, embeds carbon accounting into budget decisions through a shadow "
            "carbon price, and funds public engagement campaigns to build community understanding of "
            "the transition. The roadmap cross-references the Urban Climate Resilience Framework, the "
            "Municipal Energy Transition Plan, the Circular Economy Strategy, and the Sustainable Urban "
            "Mobility Plan as the principal delivery instruments for achieving its sectoral targets."
        ),
    },
]

# Directed relationships: (source_id, target_id, RELATION_TYPE)
EDGES: list[tuple[str, str, str]] = [
    # ── Land Use Act (doc_001) — foundational anchor ──────────────────────────
    ("doc_009", "doc_001", "IMPLEMENTS"),
    ("doc_003", "doc_001", "IMPLEMENTS"),
    ("doc_013", "doc_001", "IMPLEMENTS"),
    ("doc_029", "doc_001", "IMPLEMENTS"),
    ("doc_006", "doc_001", "SUPPLEMENTS"),
    ("doc_007", "doc_001", "REVIEWS"),
    ("doc_002", "doc_001", "REFERENCES"),
    ("doc_019", "doc_001", "REFERENCES"),
    ("doc_024", "doc_001", "REFERENCES"),
    # ── Sustainable Urban Mobility Plan (doc_002) ──────────────────────────────
    ("doc_009", "doc_002", "ALIGNS_WITH"),
    ("doc_007", "doc_002", "REFERENCES"),
    ("doc_015", "doc_002", "SUPPLEMENTS"),
    ("doc_014", "doc_002", "REFERENCES"),
    ("doc_022", "doc_002", "ALIGNS_WITH"),
    ("doc_030", "doc_002", "REFERENCES"),
    ("doc_024", "doc_002", "REFERENCES"),
    ("doc_028", "doc_002", "REFERENCES"),
    # ── Affordable Housing (doc_003) ───────────────────────────────────────────
    ("doc_009", "doc_003", "REFERENCES"),
    ("doc_011", "doc_003", "IMPLEMENTS"),
    ("doc_014", "doc_003", "REFERENCES"),
    ("doc_016", "doc_003", "REFERENCES"),
    ("doc_007", "doc_003", "REFERENCES"),
    ("doc_026", "doc_003", "ALIGNS_WITH"),
    ("doc_029", "doc_003", "ALIGNS_WITH"),
    # ── Smart City Plan (doc_004) ──────────────────────────────────────────────
    ("doc_004", "doc_012", "REQUIRES"),
    ("doc_012", "doc_004", "ENABLES"),
    ("doc_015", "doc_004", "REFERENCES"),
    ("doc_013", "doc_004", "ALIGNS_WITH"),
    ("doc_020", "doc_004", "REFERENCES"),
    ("doc_019", "doc_004", "REFERENCES"),
    ("doc_030", "doc_004", "REFERENCES"),
    ("doc_025", "doc_004", "ALIGNS_WITH"),
    # ── Urban Climate Resilience Framework (doc_005) ───────────────────────────
    ("doc_005", "doc_008", "REQUIRES"),
    ("doc_008", "doc_005", "SUPPORTS"),
    ("doc_010", "doc_005", "IMPLEMENTS"),
    ("doc_018", "doc_005", "ALIGNS_WITH"),
    ("doc_014", "doc_005", "REFERENCES"),
    ("doc_013", "doc_005", "REFERENCES"),
    ("doc_021", "doc_005", "ALIGNS_WITH"),
    ("doc_023", "doc_005", "ALIGNS_WITH"),
    ("doc_029", "doc_005", "REFERENCES"),
    ("doc_030", "doc_005", "REQUIRES"),
    # ── Public Participation Guidelines (doc_006) ──────────────────────────────
    ("doc_016", "doc_006", "RECOMMENDS_REVISION_OF"),
    ("doc_006", "doc_016", "INFORMED_BY"),
    ("doc_027", "doc_006", "REFERENCES"),
    ("doc_024", "doc_006", "REFERENCES"),
    ("doc_028", "doc_006", "REFERENCES"),
    ("doc_026", "doc_006", "REFERENCES"),
    # ── Zoning Reform Report (doc_007) ────────────────────────────────────────
    ("doc_007", "doc_003", "REFERENCES"),
    ("doc_007", "doc_002", "REFERENCES"),
    # ── Green Infrastructure Standards (doc_008) ───────────────────────────────
    ("doc_010", "doc_008", "ALIGNS_WITH"),
    ("doc_014", "doc_008", "REQUIRES"),
    ("doc_018", "doc_008", "ALIGNS_WITH"),
    ("doc_021", "doc_008", "REQUIRES"),
    ("doc_022", "doc_008", "ALIGNS_WITH"),
    ("doc_023", "doc_008", "REQUIRES"),
    ("doc_027", "doc_008", "REFERENCES"),
    # ── Transit-Oriented Development Policy (doc_009) ─────────────────────────
    ("doc_022", "doc_009", "ALIGNS_WITH"),
    # ── Heat Island Strategy (doc_010) ────────────────────────────────────────
    ("doc_022", "doc_010", "ALIGNS_WITH"),
    # ── Urban Regeneration Programme (doc_011) ────────────────────────────────
    ("doc_018", "doc_011", "REFERENCES"),
    ("doc_011", "doc_015", "REFERENCES"),
    ("doc_011", "doc_016", "ALIGNS_WITH"),
    ("doc_026", "doc_011", "REFERENCES"),
    ("doc_027", "doc_011", "REFERENCES"),
    # ── Digital Urban Infrastructure Act (doc_012) ────────────────────────────
    ("doc_015", "doc_012", "REFERENCES"),
    ("doc_019", "doc_012", "REFERENCES"),
    ("doc_020", "doc_012", "REFERENCES"),
    # ── Economic Development Zones (doc_013) ──────────────────────────────────
    ("doc_020", "doc_013", "ALIGNS_WITH"),
    ("doc_025", "doc_013", "REFERENCES"),
    # ── Waterfront Master Plan (doc_014) ──────────────────────────────────────
    ("doc_017", "doc_014", "SUPPLEMENTS"),
    ("doc_023", "doc_014", "REFERENCES"),
    # ── Air Quality Policy (doc_015) ──────────────────────────────────────────
    ("doc_029", "doc_015", "REFERENCES"),
    ("doc_030", "doc_015", "ALIGNS_WITH"),
    # ── Equity Report (doc_016) ───────────────────────────────────────────────
    ("doc_027", "doc_016", "ALIGNS_WITH"),
    ("doc_028", "doc_016", "ALIGNS_WITH"),
    ("doc_026", "doc_016", "ALIGNS_WITH"),
    # ── Historic Preservation Guidelines (doc_017) ────────────────────────────
    ("doc_025", "doc_017", "REFERENCES"),
    # ── Food Security Policy (doc_018) ────────────────────────────────────────
    ("doc_018", "doc_011", "REFERENCES"),
    # ── Noise Ordinance (doc_019) ─────────────────────────────────────────────
    ("doc_024", "doc_019", "REFERENCES"),
    ("doc_022", "doc_019", "ALIGNS_WITH"),
    # ── Circular Economy / Waste (doc_020) ────────────────────────────────────
    ("doc_030", "doc_020", "REQUIRES"),
    ("doc_018", "doc_020", "ALIGNS_WITH"),
    # ── Integrated Water Management (doc_021) ─────────────────────────────────
    ("doc_029", "doc_021", "REFERENCES"),
    # ── Walkable Neighbourhoods (doc_022) ─────────────────────────────────────
    ("doc_028", "doc_022", "ALIGNS_WITH"),
    ("doc_027", "doc_022", "ALIGNS_WITH"),
    # ── Urban Biodiversity Plan (doc_023) ─────────────────────────────────────
    ("doc_023", "doc_021", "ALIGNS_WITH"),
    # ── Night-Time Economy (doc_024) ──────────────────────────────────────────
    ("doc_024", "doc_020", "REFERENCES"),
    # ── Energy Transition Plan (doc_025) ──────────────────────────────────────
    ("doc_025", "doc_030", "IMPLEMENTS"),
    # ── Housing First (doc_026) ───────────────────────────────────────────────
    ("doc_026", "doc_029", "REFERENCES"),
    # ── Net-Zero Roadmap (doc_030) ────────────────────────────────────────────
    ("doc_030", "doc_025", "REQUIRES"),
    ("doc_030", "doc_008", "REFERENCES"),
]

# ── Graph path explanations (static — illustrative paths for the demo query) ──

_GRAPH_PATHS: dict[str, str] = {
    # ── Example query: "reducing urban heat through green infrastructure" ───────
    # Dense seeds: doc_010 (Heat Island), doc_008 (Green Infrastructure), doc_005 (Climate Resilience)
    # 1-hop neighbours of those seeds:
    "doc_021": "hop=1 from doc_008 via REQUIRES (Water Management)",
    "doc_023": "hop=1 from doc_008 via REQUIRES (Biodiversity Plan)",
    "doc_018": "hop=1 from doc_005 via ALIGNS_WITH (Food Security Policy)",
    "doc_013": "hop=1 from doc_005 via REFERENCES (Economic Development Zones)",
    "doc_014": "hop=1 from doc_008 via REQUIRES (Waterfront Master Plan)",
    "doc_022": "hop=1 from doc_008 via ALIGNS_WITH (Walkable Streets)",
    "doc_029": "hop=1 from doc_005 via REFERENCES (Brownfield Regeneration)",
    "doc_030": "hop=1 from doc_005 via REQUIRES (Net-Zero Roadmap)",
    # 2-hop neighbours:
    "doc_004": "hop=2 from doc_013 via ALIGNS_WITH → Smart City Plan",
    "doc_002": "hop=2 from doc_014 via REFERENCES → Sustainable Mobility Plan",
    "doc_003": "hop=2 from doc_014 via REFERENCES → Affordable Housing Framework",
    "doc_025": "hop=2 from doc_030 via REQUIRES → Energy Transition Plan",
    "doc_015": "hop=2 from doc_030 via ALIGNS_WITH → Air Quality Policy",
    "doc_020": "hop=2 from doc_030 via REQUIRES → Circular Economy Strategy",
    "doc_011": "hop=2 from doc_018 via REFERENCES → Urban Regeneration Programme",
    "doc_027": "hop=2 from doc_022 via ALIGNS_WITH → Child Inclusive Design Guidelines",
    "doc_017": "hop=2 from doc_014 via SUPPLEMENTS → Historic Preservation Guidelines",
    "doc_028": "hop=2 from doc_022 via ALIGNS_WITH → Age-Friendly City Plan",
}


# ── Setup helpers ──────────────────────────────────────────────────────────────

async def _build_graph(graph: Neo4jGraph) -> None:
    """Upsert Document nodes and edges in Neo4j (MERGE — safe to re-run)."""
    print("\n[GRAPH] Upserting document nodes ...")
    for doc in DOCUMENTS:
        await graph.upsert_node(
            doc["id"],
            {"title": doc["title"], "doc_type": doc["doc_type"]},
        )
        print(f"  MERGE :Document {{id: {doc['id']!r}}}")

    print("[GRAPH] Upserting edges ...")
    for src, tgt, rel in EDGES:
        await graph.upsert_edge(src, tgt, rel)
        print(f"  MERGE ({src})-[:{rel}]->({tgt})")


async def _populate_docstore(docstore: DocStoreBackend) -> None:
    """Add one document + one chunk per document (upserts on re-run)."""
    print("\n[DOCSTORE] Adding documents and chunks ...")
    for doc in DOCUMENTS:
        await docstore.add_document(
            id=doc["id"],
            content=doc["content"],
            doc_type=doc["doc_type"],
            metadata={"title": doc["title"]},
        )
        # Chunk id equals document id — simplifies result display for this demo.
        await docstore.add_chunk(
            id=doc["id"],
            document_id=doc["id"],
            content=doc["content"],
            chunk_index=0,
            metadata={"title": doc["title"], "doc_type": doc["doc_type"]},
        )
        print(f"  + {doc['id']}: {doc['title'][:55]}")


def _build_index(dim: int, vectors: np.ndarray, doc_ids: list[str]) -> QuantaIndex:
    """Create a QuantaIndex from precomputed embeddings."""
    print(f"\n[INDEX] Building vector index (dim={dim}, bit_width=4) ...")
    idx = QuantaIndex(name="legal_text", dim=dim, bit_width=4)
    idx.add(vectors, doc_ids)
    print(f"  Indexed {len(idx)} vectors")
    return idx


# ── Output helpers ─────────────────────────────────────────────────────────────

def _sep(text: str = "") -> None:
    width = max(len(text) + 2, 52)
    print("─" * width)
    if text:
        print(f" {text}")
        print("─" * width)


def _print_results(label: str, results: list[RetrievalResult]) -> None:
    print(f"\n[{label}]")
    for rank, r in enumerate(results, 1):
        title = r.metadata.get("title", r.id)
        source_col = f"source={r.source}"
        print(f"  {rank}. {r.id:<10}  score={r.score:.3f}  {source_col:<22}  {title}")


def _print_comparison(
    results_a: list[RetrievalResult],
    results_b: list[RetrievalResult],
) -> None:
    ids_a = {r.id for r in results_a}
    ids_b = {r.id for r in results_b}

    newly_found = ids_b - ids_a
    graph_confirmed = {r.id for r in results_b if "graph" in r.source}
    highlighted = newly_found | graph_confirmed

    print("\nGraph expansion surfaced:")
    if not highlighted:
        print("  (no additional documents reached via graph traversal)")
        return

    # Show newly found first, then confirmed
    for doc_id in sorted(highlighted, key=lambda x: (x not in newly_found, x)):
        title = next((d["title"] for d in DOCUMENTS if d["id"] == doc_id), doc_id)
        path = _GRAPH_PATHS.get(doc_id, "reached via graph traversal")
        tag = " [NEW]" if doc_id in newly_found else ""
        print(f"  • {doc_id} ({title}){tag}: {path}")


# ── Cleanup ────────────────────────────────────────────────────────────────────

async def _cleanup(graph: Neo4jGraph, docstore: DocStoreBackend) -> None:
    print("\n[CLEANUP] Removing test data ...")
    for doc in DOCUMENTS:
        await graph.delete_node(doc["id"])
        await docstore.delete_document(doc["id"])
    print("  Done — Neo4j nodes and docstore records removed.")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    # ── 1. SETUP ──────────────────────────────────────────────────────────────
    load_dotenv()

    # QuantaSettings validates POSTGRES_* even when DOCSTORE_BACKEND=duckdb.
    # setdefault avoids overriding real credentials that may already be set.
    os.environ.setdefault("POSTGRES_USER", "_unused_")
    os.environ.setdefault("POSTGRES_PASSWORD", "_unused_")

    cfg = get_settings()
    run_cleanup = os.environ.get("RUN_CLEANUP", "false").lower() == "true"

    print(f"[EMBED] Loading model: {cfg.EMBED_MODEL}")
    st = SentenceTransformer(cfg.EMBED_MODEL)

    def embed(texts: list[str]) -> np.ndarray:
        return st.encode(texts, normalize_embeddings=True).astype(np.float32)

    if not cfg.graph_configured:
        print(
            "ERROR: Neo4j is not configured.\n"
            "  Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in your .env file."
        )
        sys.exit(1)

    try:
        graph = Neo4jGraph(
            uri=cfg.NEO4J_URI,           # type: ignore[arg-type]
            user=cfg.NEO4J_USER,         # type: ignore[arg-type]
            password=cfg.NEO4J_PASSWORD, # type: ignore[arg-type]
            database=cfg.NEO4J_DATABASE,
        )
    except Exception as exc:
        msg = str(exc)
        if "neo4j driver" in msg or "pip install" in msg:
            print(f"Missing dependency: {exc}\n  Run: pip install quanta[neo4j]")
        else:
            print(f"Could not connect to Neo4j. Is it running? Check NEO4J_URI\n  Detail: {exc}")
        sys.exit(1)

    try:
        await graph._driver.verify_connectivity()
        print(f"[SETUP] Connected to Neo4j at {cfg.NEO4J_URI}")
    except Exception as exc:
        print(f"Could not connect to Neo4j. Is it running? Check NEO4J_URI\n  Detail: {exc}")
        await graph.close()
        sys.exit(1)

    docstore = get_docstore(cfg)
    await docstore.init()
    print(f"[SETUP] Connected to docstore ({cfg.DOCSTORE_BACKEND}: {cfg.DUCKDB_PATH})")

    idx: QuantaIndex | None = None

    try:
        # ── 2. BUILD THE GRAPH ─────────────────────────────────────────────────
        await _build_graph(graph)

        # ── 3. ADD DOCUMENTS TO DOCSTORE ──────────────────────────────────────
        await _populate_docstore(docstore)

        # ── 4. CREATE VECTOR INDEX ─────────────────────────────────────────────
        doc_ids = [d["id"] for d in DOCUMENTS]
        vectors = embed([d["content"] for d in DOCUMENTS])
        idx = _build_index(cfg.EMBED_DIM, vectors, doc_ids)

        # ── 5 & 6. QUERIES AND RESULTS ─────────────────────────────────────────
        retriever = MultiRetriever(
            indexes={"legal_text": idx},
            docstore=docstore,
            graph=graph,
        )
        print("[INFO] MultiRetriever initialized with vector index and graph.")
        
        query_text = "How can the city reduce urban heat island effects through green infrastructure and tree canopy?"
        # query_text = "What policies ensure affordable housing is located near public transport stations?",
        # query_text = "What is the city's plan to decarbonise buildings, transport, and energy supply by 2045?",
        # query_text = "How are IoT sensors and digital platforms used to monitor and manage urban services?",
        # query_text = "How does the city ensure that planning decisions do not disadvantage low-income communities?",

        query_vec = embed([query_text])[0]

        print()
        _sep(f'QUERY: "{query_text}"')

        # Query A — dense vector search only
        results_a = await retriever.search(
            query_vectors={"legal_text": query_vec},
            k=5,
            use_graph=False,
        )
        _print_results("A] Dense-only results", results_a)

        # Query B — dense search + 2-hop graph expansion from top-3 seeds
        results_b = await retriever.search(
            query_vectors={"legal_text": query_vec},
            k=5,
            use_graph=True,
            graph_hops=2,
            graph_seed_k=3,
        )
        _print_results("B] Dense + Graph results", results_b)

        _print_comparison(results_a, results_b)

    finally:
        # ── 7. CLEANUP (optional) ──────────────────────────────────────────────
        if run_cleanup and idx is not None:
            await _cleanup(graph, docstore)
        elif not run_cleanup:
            print(
                "\n[INFO] Set RUN_CLEANUP=true in .env to remove test data on the next run."
            )

        await graph.close()
        await docstore.close()


if __name__ == "__main__":
    asyncio.run(main())
