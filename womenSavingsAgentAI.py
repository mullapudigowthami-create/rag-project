from crewai import Agent, Task, Crew, LLM

llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

# ============================================================
# CUSTOMER DATA
# ============================================================

customer_data = """customer_id,name,customer_type,monthly_salary,deposit_amount,deposit_frequency,lock_in_years,account_status
C001,Priya Sharma,Working Woman,45000,5000,Monthly,5,Active
C002,Lakshmi Devi,Homemaker,0,500,Weekly,3,Active
C003,Sunita Rao,Working Woman,12000,1000,Monthly,10,Active
C004,Meena Kumari,Homemaker,0,2000,Monthly,5,Active
C005,Radha Pillai,Working Woman,75000,10000,Monthly,10,Active
C006,Anitha Nair,Homemaker,0,200,Irregular,3,Active
C007,Kavitha Reddy,Working Woman,22000,3000,Monthly,5,Active
C008,Fatima Begum,Homemaker,0,5000,Monthly,10,Active
C009,Deepa Singh,Working Woman,8000,500,Monthly,3,Active
C010,Saroja Devi,Homemaker,0,100,Weekly,5,Active
"""

salary_slabs = """salary_range,daily_deduction_salary,daily_deduction_deposit_small,daily_deduction_deposit_medium,daily_deduction_deposit_large
Below 10000,5,5,50,100
10000-25000,50,5,50,100
25000-50000,100,50,100,500
Above 50000,500,50,100,500
"""

deposit_slabs = """deposit_type,deposit_range,daily_deduction
Small,Below 1000,5
Medium,1000-5000,50
Above Average,5000-10000,100
Large,Above 10000,500
"""

withdrawal_cases = """case_id,customer_id,customer_name,reason,evidence_provided,years_completed,request_status
W001,C003,Sunita Rao,Medical Emergency,Hospital Bills and Doctor Certificate,2,Pending
W002,C007,Kavitha Reddy,No valid reason,None,1,Rejected
W003,C001,Priya Sharma,Family Emergency,Police Report and Documents,4,Pending
"""

# Save CSV files
with open("customers.csv", "w") as f:
    f.write(customer_data)
with open("salary_slabs.csv", "w") as f:
    f.write(salary_slabs)
with open("deposit_slabs.csv", "w") as f:
    f.write(deposit_slabs)
with open("withdrawal_cases.csv", "w") as f:
    f.write(withdrawal_cases)

# ============================================================
# DEFINE 5 AGENTS
# ============================================================

customer_profile_agent = Agent(
    role="Customer Profile Specialist",
    goal="Identify each customer type and assign correct deduction category",
    backstory="""You are a bank customer profiling specialist for the Women and Family 
    Savings Scheme in India. You identify if a customer is a working woman or homemaker 
    and assign the correct salary slab or deposit category.
    
    Working Women rules:
    - Below Rs 10,000 salary -> Minimum slab -> Rs 5/day salary deduction
    - Rs 10,000-25,000 -> Average slab -> Rs 50/day salary deduction
    - Rs 25,000-50,000 -> Above average -> Rs 100/day salary deduction
    - Above Rs 50,000 -> High salary -> Rs 500/day salary deduction
    - Working women get BOTH salary-based AND deposit-based deductions combined
    
    Homemaker rules:
    - Small deposit (below Rs 1,000) -> Rs 5/day
    - Medium deposit (Rs 1,000-5,000) -> Rs 50/day
    - Above average (Rs 5,000-10,000) -> Rs 100/day
    - Large deposit (above Rs 10,000) -> Rs 500/day
    - Homemakers get deposit-based deduction only""",
    llm=llm,
    verbose=True
)

savings_calculator_agent = Agent(
    role="Savings Calculator Agent",
    goal="Calculate daily deductions and monthly savings for all customers",
    backstory="""You are a precise savings calculator for the Women and Family Savings 
    Scheme. You calculate:
    - Daily deduction amount for each customer
    - Monthly savings (daily deduction x 30 days)
    - For working women: salary deduction + deposit deduction combined daily
    - For homemakers: deposit deduction only
    - Interest rate: 4% per year on total savings
    - Monthly interest = total savings x 4% / 12""",
    llm=llm,
    verbose=True
)

profit_projection_agent = Agent(
    role="Profit Projection Agent",
    goal="Calculate and show profit projections for 1 month, 3 years, 5 years and 10 years",
    backstory="""You are a financial projection specialist for the Women and Family 
    Savings Scheme. For every customer you calculate profit projections at:
    - 1 Month
    - 1 Year
    - 3 Years (minimum lock-in)
    - 5 Years
    - 10 Years (maximum lock-in)
    
    Formula:
    - Monthly savings = daily deduction x 30
    - Annual savings = monthly savings x 12
    - Interest = compound interest at 4% per year
    - Total profit = total saved + total interest earned
    
    Show projections for both working women and homemakers separately.
    Always encourage customers by showing how much they will save over time.""",
    llm=llm,
    verbose=True
)

lockin_policy_agent = Agent(
    role="Lock-in and Withdrawal Policy Agent",
    goal="Manage lock-in rules, verify emergency withdrawals and handle account closures",
    backstory="""You are a strict bank policy officer for the Women and Family Savings 
    Scheme. You enforce these rules:
    
    Lock-in rules:
    - Minimum lock-in period: 3 years
    - Maximum lock-in period: 10 years
    - Amount CANNOT be withdrawn during lock-in period
    
    Emergency withdrawal rules:
    - Customer must provide valid reason
    - Evidence must be submitted (medical bills, police report etc)
    - Bank verifies the emergency
    - Only genuine emergencies are approved
    - Rejected if no valid reason or evidence
    
    Account closure rules:
    - After withdrawal account is PERMANENTLY closed
    - All customer identities are removed from system
    - Customer cannot reopen same account
    - Must open fresh new account if needed
    
    Check all withdrawal cases and approve or reject based on rules.""",
    llm=llm,
    verbose=True
)

report_writer_agent = Agent(
    role="Monthly Report Writer",
    goal="Generate complete monthly savings report for all customers with profit projections and encouragement",
    backstory="""You are a friendly and professional bank report writer for the Women 
    and Family Savings Scheme. You generate beautiful monthly reports for each customer 
    showing:
    - Customer name and type (working woman or homemaker)
    - Daily deduction amount
    - Total saved this month
    - Interest earned this month
    - Profit projections for 3 years, 5 years and 10 years
    - Lock-in period remaining
    - Encouraging message to keep saving
    - Warning about withdrawal consequences
    
    Make the report warm, encouraging and easy to understand for Indian families.""",
    llm=llm,
    verbose=True
)

# ============================================================
# DEFINE 5 TASKS
# ============================================================

task1 = Task(
    description=f"""Analyze each customer and identify their type and correct deduction category:
    {customer_data}
    
    Salary slabs: {salary_slabs}
    Deposit slabs: {deposit_slabs}
    
    For each customer:
    1. Identify if Working Woman or Homemaker
    2. Assign correct salary slab (working women) or deposit slab (homemakers)
    3. For working women apply BOTH salary + deposit deduction rules
    4. For homemakers apply deposit deduction only
    5. State daily deduction amount for each customer""",
    expected_output="Complete customer profile report with deduction categories for all customers",
    agent=customer_profile_agent
)

task2 = Task(
    description=f"""Calculate daily and monthly savings for all customers:
    {customer_data}
    
    Rules:
    - Working women: salary deduction + deposit deduction = total daily deduction
    - Homemakers: deposit deduction only
    - Monthly savings = daily deduction x 30 days
    - Monthly interest = total savings x 4% / 12
    - Total monthly profit = monthly savings + monthly interest
    
    Calculate for each customer and show breakdown clearly.""",
    expected_output="Detailed daily and monthly savings calculation for all customers",
    agent=savings_calculator_agent
)

task3 = Task(
    description=f"""Calculate profit projections for all customers:
    {customer_data}
    
    Show projections for each customer at:
    - 1 Month
    - 1 Year  
    - 3 Years (minimum lock-in)
    - 5 Years
    - 10 Years (maximum lock-in)
    
    Use 4% annual compound interest.
    Show projections separately for:
    - All working women customers
    - All homemaker customers
    
    Always highlight how much MORE they earn by staying for 10 years vs 3 years
    to encourage longer savings.""",
    expected_output="Complete profit projection tables for all customers for 1 month, 3, 5 and 10 years",
    agent=profit_projection_agent
)

task4 = Task(
    description=f"""Review all withdrawal requests and apply lock-in policy:
    {withdrawal_cases}
    {customer_data}
    
    For each withdrawal case:
    1. Check if lock-in period minimum (3 years) is completed
    2. Check if valid reason is provided
    3. Check if evidence is submitted
    4. Approve or Reject with clear reason
    5. For approved cases: mark account for permanent closure
    6. For rejected cases: explain why and remind of lock-in rules
    
    Also flag any customers who are close to completing their lock-in period.""",
    expected_output="Complete withdrawal case decisions with approvals, rejections and account closure notices",
    agent=lockin_policy_agent
)

task5 = Task(
    description="""Write a complete WOMEN AND FAMILY SAVINGS SCHEME MONTHLY REPORT combining all findings:

    Include these sections:
    1. Scheme Overview
       - Who can join: Working Women and Homemakers
       - Indian families focused
       - Small government and private banks
    
    2. Customer Summary
       - Total working women enrolled
       - Total homemakers enrolled
       - Total daily savings across all customers
    
    3. Deduction Rules Summary
       - Working women: salary + deposit based (both combined)
       - Homemakers: deposit based only
       - Daily deduction slabs clearly explained
    
    4. Monthly Savings Report
       - Each customer savings this month
       - Interest earned
       - Total profit
    
    5. Profit Projections
       - Show 1 month, 3 year, 5 year, 10 year projections
       - For both working women and homemakers
    
    6. Lock-in Policy Reminder
       - Minimum 3 years, maximum 10 years
       - Cannot withdraw without emergency
       - Evidence required for emergency withdrawal
       - Account permanently closed after withdrawal
    
    7. Withdrawal Case Updates
       - Approved cases with next steps
       - Rejected cases with reasons
    
    8. Encouragement Message
       - Motivate all customers especially homemakers
       - Show power of small daily savings over 10 years
       - Remind them this scheme protects them in hard times
    
    Make it warm, professional and easy to understand for Indian families!""",
    expected_output="Complete Women and Family Savings Scheme Monthly Report",
    agent=report_writer_agent
)

# ============================================================
# CREATE AND RUN THE CREW
# ============================================================

crew = Crew(
    agents=[
        customer_profile_agent,
        savings_calculator_agent,
        profit_projection_agent,
        lockin_policy_agent,
        report_writer_agent
    ],
    tasks=[task1, task2, task3, task4, task5],
    verbose=True
)

print("*** Women and Family Savings Scheme AI Started! ***")
print("*** Running 5 Agents - Profile, Calculator, Projections, Policy, Report ***")
print("=" * 60)

result = crew.kickoff()

print("\n" + "=" * 60)
print("*** FINAL WOMEN AND FAMILY SAVINGS SCHEME REPORT ***")
print("=" * 60)
print(result)