from crewai import Agent, Task, Crew, LLM

llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

# Climate Data
nature_data = """year,avg_temp_celsius,rainfall_mm,forest_cover_percent,air_quality_index,co2_ppm
2018,14.2,820,31.2,85,408.5
2019,14.8,790,30.8,88,411.4
2020,14.5,850,30.5,72,413.9
2021,15.1,760,30.1,90,416.2
2022,15.4,730,29.7,95,418.8
2023,15.8,700,29.2,98,421.1
"""

# Housing Data
housing_data = """house_id,owner,trees_count,tree_types,tree_age_years,distance_from_house_meters,near_pole_or_cctv,country,land_type
H001,Ramesh,3,"Neem,Teak,Banyan",12,10,No,India,Normal
H002,Suresh,1,"Neem",5,8,No,India,Normal
H003,Priya,5,"Neem,Peepal,Teak,Banyan,Neem",15,12,No,India,Normal
H004,Vikram,2,"Teak,Peepal",3,6,Yes,India,Normal
H005,John,4,"Neem,Banyan,Teak,Peepal",11,15,No,USA,Normal
H006,Maria,0,"None",0,0,No,Brazil,Dry
H007,Chen,2,"Teak,Neem",7,9,No,China,Normal
H008,Lakshmi,3,"Banyan,Peepal,Neem",18,11,No,India,Normal
H009,David,1,"Teak",2,5,Yes,UK,Normal
H010,Fatima,4,"Neem,Banyan,Peepal,Teak",13,14,No,India,Normal
H011,Carlos,3,"Neem,Teak,Banyan",10,10,No,Brazil,Dry
H012,Sophie,2,"Peepal,Neem",11,12,No,France,Normal
"""

# Tree Value Data
tree_value_data = """tree_type,growth_rate,timber_value_10yr_inr,medicinal_value,temperature_reduction_celsius,maintenance_level,recommended_distance_meters
Neem,Fast,25000,Very High,2.5,Low,8
Teak,Medium,150000,Medium,2.0,Low,10
Banyan,Slow,50000,High,3.5,Low,12
Peepal,Medium,35000,Very High,3.0,Low,10
"""

# Incentive Data
incentive_data = """country,currency,incentive_10plus_years,incentive_15plus_years,incentive_20plus_years
India,INR,500,750,1000
USA,USD,20,30,50
UK,GBP,15,25,40
Brazil,BRL,100,150,200
China,CNY,100,150,200
France,EUR,18,28,45
"""

# Tree Cutting Violations Data
cutting_violations = """house_id,owner,trees_cut,tree_age_cut,country,current_tree_count
V001,Ramesh,1,12,India,3
V002,Suresh,1,5,India,1
V003,Carlos,2,10,Brazil,3
V004,Sophie,1,11,France,2
"""

# Save all CSV files
with open("nature_climate.csv", "w") as f:
    f.write(nature_data)
with open("housing_trees.csv", "w") as f:
    f.write(housing_data)
with open("tree_values.csv", "w") as f:
    f.write(tree_value_data)
with open("green_incentives.csv", "w") as f:
    f.write(incentive_data)
with open("cutting_violations.csv", "w") as f:
    f.write(cutting_violations)

# ============================================================
# DEFINE ALL 6 AGENTS
# ============================================================

environmental_analyst = Agent(
    role="Environmental Analyst",
    goal="Analyze climate data and show urgent need for trees worldwide",
    backstory="""You are a senior environmental scientist with 20 years experience.
    You analyze climate data showing temperature rise, CO2 increase and rainfall decrease.
    You strongly advocate that every house must have 2 to 5 trees to fight climate change.""",
    llm=llm,
    verbose=True
)

compliance_agent = Agent(
    role="Housing Tree Compliance Officer",
    goal="Check every house for tree count compliance and identify violations",
    backstory="""You are a strict government green policy compliance officer.
    Rules you enforce:
    1. Every house MUST have minimum 2 trees and maximum 5 trees
    2. Trees must be planted minimum 8 meters away from house - cluster free
    3. Trees must NOT be near electricity poles or CCTV cameras
    4. Houses with 0 trees are critically non-compliant
    5. Dry land houses must plant trees one month before rainy season
    6. Every citizen must water their trees minimum once a week""",
    llm=llm,
    verbose=True
)

tree_recommendation_agent = Agent(
    role="Tree Recommendation Specialist",
    goal="Recommend best trees and planting guidelines for every household",
    backstory="""You are a forestry expert recommending best trees for homes.
    Top trees: Neem, Teak, Banyan, Peepal.
    For dry land: plant 1 month before rainy season so rain helps roots grow naturally.
    After rainy season: citizen responsibility to water once a week minimum.
    Trees must be cluster free and away from poles and CCTV cameras.""",
    llm=llm,
    verbose=True
)

incentive_agent = Agent(
    role="Green Incentive Calculator",
    goal="Calculate monthly government payments for trees aged 10 years and above",
    backstory="""You calculate government green incentive payments.
    Only trees aged 10 years and ABOVE qualify for monthly payment.
    Trees UNDER 10 years do not qualify yet.
    India: Rs 500/month (10+yrs), Rs 750 (15+yrs), Rs 1000 (20+yrs)
    USA: $20 (10+), $30 (15+), $50 (20+)
    UK: £15 (10+), £25 (15+), £40 (20+)
    Brazil: R$100 (10+), R$150 (15+), R$200 (20+)
    China: ¥100 (10+), ¥150 (15+), ¥200 (20+)
    France: €18 (10+), €28 (15+), €45 (20+)""",
    llm=llm,
    verbose=True
)

law_enforcement_agent = Agent(
    role="Green Law Enforcement Officer",
    goal="Enforce tree cutting laws and calculate fines for violations",
    backstory="""You enforce strict tree cutting laws:
    1. Cutting ANY tree is a PUNISHABLE OFFENCE
    2. Fine = DOUBLE the monthly incentive of that tree
       Example: Tree earns Rs 500/month -> fine = Rs 1000
    3. Cutting ONLY allowed if house has MAXIMUM 5 trees
    4. Cutting STRICTLY PROHIBITED if house has minimum 2 trees or below
    5. Dry barren land tree cutting = most serious offence
    6. Repeat offenders face double fines
    Check violations and calculate fines strictly.""",
    llm=llm,
    verbose=True
)

policy_writer = Agent(
    role="Global Green Policy Report Writer",
    goal="Write a powerful comprehensive green policy report for governments worldwide",
    backstory="""You are a senior government policy writer creating actionable
    environmental policy reports combining all agent findings into one powerful document.""",
    llm=llm,
    verbose=True
)

# ============================================================
# DEFINE ALL 6 TASKS
# ============================================================

task1 = Task(
    description=f"""Analyze this climate data:
    {nature_data}
    Show temperature rise, CO2 increase and rainfall decrease trends.
    Explain how 2-5 trees per house nationwide can significantly reduce these problems.""",
    expected_output="Climate analysis showing urgent need for residential trees",
    agent=environmental_analyst
)

task2 = Task(
    description=f"""Check each house for green policy compliance:
    {housing_data}

    Rules:
    1. Minimum 2 trees, Maximum 5 trees per house
    2. Trees minimum 8 meters from house - cluster free
    3. No trees near electricity poles or CCTV cameras
    4. Houses with 0 trees = critical violation - must plant immediately
    5. Dry land houses must plant 1 month before rainy season
    6. Every citizen must water trees minimum once a week

    List compliant, non-compliant houses and required actions.""",
    expected_output="Full housing compliance report with required actions",
    agent=compliance_agent
)

task3 = Task(
    description=f"""Recommend best trees based on:
    {tree_value_data}

    Include:
    1. Neem, Teak, Banyan, Peepal - full benefits of each
    2. Planting distance from house - cluster free
    3. Keep away from electricity poles and CCTV cameras
    4. Dry land rule: plant 1 month BEFORE rainy season
       - Rain helps trees establish roots naturally
       - Reduces citizen workload after rainy season
    5. Citizen responsibility: water trees once a week minimum
    6. Trees grow in value - timber and medicinal benefits over time""",
    expected_output="Complete tree recommendation and planting guidelines",
    agent=tree_recommendation_agent
)

task4 = Task(
    description=f"""Calculate government incentive payments:
    Housing: {housing_data}
    Rates: {incentive_data}

    Only trees 10 years and above qualify.
    Calculate monthly payment per tree and total per household.
    For trees under 10 years show when they will qualify.
    Show all amounts in local currency per country.""",
    expected_output="Monthly incentive payment report for all qualifying households",
    agent=incentive_agent
)

task5 = Task(
    description=f"""Enforce tree cutting laws:
    Violations: {cutting_violations}
    Incentive rates: {incentive_data}

    For each violation:
    1. Check if cutting was LEGAL - only allowed at maximum 5 trees
    2. If house had minimum 2 trees or less - ILLEGAL, maximum punishment
    3. Calculate fine = DOUBLE the monthly incentive of cut tree
    4. State legal or illegal offence
    5. Recommend punishment for each violator
    6. Dry land cutting = most serious offence""",
    expected_output="Complete violation report with fines and punishments per violator",
    agent=law_enforcement_agent
)

task6 = Task(
    description="""Write GLOBAL GREEN POLICY REPORT with these sections:

    1. Executive Summary - Green crisis and urgent action needed
    2. Climate Data - Temperature, CO2 and rainfall trends
    3. Housing Tree Policy - Every house must have 2 to 5 trees
    4. Recommended Trees - Neem, Teak, Banyan, Peepal benefits
    5. Planting Guidelines:
       - Cluster free, away from house poles and CCTV
       - Dry land: plant 1 month before rainy season
       - Citizen duty: water trees once a week minimum
       - After rainy season: citizen responsibility to maintain
    6. Government Incentive Scheme:
       - Trees 10 years and above get monthly payment
       - Rates by country in local currency
    7. Tree Cutting Laws:
       - Cutting = punishable offence
       - Fine = double the monthly incentive
       - Only allowed at maximum tree count (5)
       - Prohibited at minimum tree count (2)
    8. Global Call to Action

    Make it powerful and ready for government presentation worldwide!""",
    expected_output="Complete Global Green Policy Report for government presentation",
    agent=policy_writer
)

# ============================================================
# CREATE AND RUN THE CREW
# ============================================================

crew = Crew(
    agents=[
        environmental_analyst,
        compliance_agent,
        tree_recommendation_agent,
        incentive_agent,
        law_enforcement_agent,
        policy_writer
    ],
    tasks=[task1, task2, task3, task4, task5, task6],
    verbose=True
)

print("*** Mother Nature Agent AI Started!")
print("*** Running 6 Agents - Climate, Compliance, Trees, Incentives, Law, Report...")
print("=" * 60)

result = crew.kickoff()

print("\n" + "=" * 60)
print("*** FINAL GLOBAL GREEN POLICY REPORT ***")
print("=" * 60)
print(result)