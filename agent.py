import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
import csv
from io import StringIO

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

logger = logging.getLogger(__name__)

APP_NAME = "mcp_financial_coach"
USER_ID = "user_michael"

class SpendingCategory(BaseModel):
    category: str = Field(..., description="Expense category name")
    amount: float = Field(..., description="Amount spent in this category")
    percentage: Optional[float] = Field(None, description="Percentage of total spending")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Category for recommendation")
    recommendation: str = Field(..., description="Recommendation details")
    potential_savings: Optional[float] = Field(None, description="Estimated monthly savings")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total monthly expenses")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    spending_categories: List[SpendingCategory] = Field(..., description="Breakdown of spending by category")
    recommendations: List[SpendingRecommendation] = Field(..., description="Spending recommendations")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Recommended emergency fund size")
    current_amount: Optional[float] = Field(None, description="Current emergency fund (if any)")
    current_status: str = Field(..., description="Status assessment of emergency fund")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Savings category")
    amount: float = Field(..., description="Recommended monthly amount")
    rationale: Optional[str] = Field(None, description="Explanation for this recommendation")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Name of automation technique")
    description: str = Field(..., description="Details of how to implement")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Emergency fund recommendation")
    recommendations: List[SavingsRecommendation] = Field(..., description="Savings allocation recommendations")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Automation techniques to help save")

class Debt(BaseModel):
    name: str = Field(..., description="Name of debt")
    amount: float = Field(..., description="Current balance")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    min_payment: Optional[float] = Field(None, description="Minimum monthly payment")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total interest paid")
    months_to_payoff: int = Field(..., description="Months until debt-free")
    monthly_payment: Optional[float] = Field(None, description="Recommended monthly payment")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Highest interest first method")
    snowball: PayoffPlan = Field(..., description="Smallest balance first method")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Title of recommendation")
    description: str = Field(..., description="Details of recommendation")
    impact: Optional[str] = Field(None, description="Expected impact of this action")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total debt amount")
    debts: List[Debt] = Field(..., description="List of all debts")
    payoff_plans: PayoffPlans = Field(..., description="Debt payoff strategies")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Recommendations for debt reduction")

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def get_finance_advisor_system():
    return FinanceAdvisorSystem()

def parse_json_safely(data: str, default_value: Any = None) -> Any:
    """
    Safely parses a JSON string into a Python object.
    If the input is not a string or if JSON decoding fails, it returns a default value.
    """
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value

def _render_sidebar_content_as_column():
    """
    Renders the content for the Streamlit sidebar, including setup instructions and CSV template download.
    """
    st.header("Setup & Templates")
    st.info("Please ensure you have your Gemini API key in the .env file:\n```\nGOOGLE_API_KEY=your_api_key_here\n```")
    st.caption("This application uses Google's ADK (Agent Development Kit) and Gemini AI to provide personalized financial advice.")
    
   
    
    # Add CSV template download
    st.subheader("CSV Template")
    st.markdown("""
        Download the template CSV file with the required format:
        - Date (YYYY-MM-DD)
        - Category
        - Amount (numeric)
        """)
    
    # Create sample CSV content
    sample_csv = """Date,Category,Amount
        2024-01-01,Housing,1200.00
        2024-01-02,Food,150.50
        2024-01-03,Transportation,45.00"""
    
    st.download_button(
        label="Download CSV Template",
        data=sample_csv,
        file_name="expense_template.csv",
        mime="text/csv",
        key="sidebar_initial_template"
    )

class FinanceAdvisorSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()
        
        self.budget_analysis_agent = LlmAgent(
            name="BudgetAnalysisAgent",
            model="gemini-2.0-flash-exp",
            description="Analyzes financial data to categorize spending patterns and recommend budget improvements",
            instruction="You are a Budget Analysis Agent specialized in reviewing financial transactions and expenses.\nYou are the first agent in a sequence of three financial advisor agents.\n\nYour tasks:\n1. Analyze income, transactions, and expenses in detail\n2. Categorize spending into logical groups with clear breakdown\n3. Identify spending patterns and trends across categories\n4. Suggest specific areas where spending could be reduced with concrete suggestions\n5. Provide actionable recommendations with specific, quantified potential savings amounts\n\nConsider:\n- Number of dependants when evaluating household expenses\n- Typical spending ratios for the income level (housing 30%, food 15%, etc.)\n- Essential vs discretionary spending with clear separation\n- Seasonal spending patterns if data spans multiple months\n\nFor spending categories, include ALL expenses from the user's data, ensure percentages add up to 100%,\nand make sure every expense is categorized.\n\nFor recommendations:\n- Provide at least 3-5 specific, actionable recommendations with estimated savings\n- Explain the reasoning behind each recommendation\n- Consider the impact on quality of life and long-term financial health\n- Suggest specific implementation steps for each recommendation\n\nIMPORTANT: Store your analysis in state['budget_analysis'] for use by subsequent agents.",
            output_schema=BudgetAnalysis,
            output_key="budget_analysis",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True
        )
        
        self.savings_strategy_agent = LlmAgent(
            name="SavingsStrategyAgent",
            model="gemini-2.0-flash-exp",
            description="Recommends optimal savings strategies based on income, expenses, and financial goals",
            instruction="You are a Savings Strategy Agent specialized in creating personalized savings plans.\nYou are the second agent in the sequence. READ the budget analysis from state['budget_analysis'] first.\n\nYour tasks:\n1. Review the budget analysis results from state['budget_analysis']\n2. Recommend comprehensive savings strategies based on the analysis\n3. Calculate optimal emergency fund size based on expenses and dependants\n4. Suggest appropriate savings allocation across different purposes\n5. Recommend practical automation techniques for saving consistently\n\nConsider:\n- Risk factors based on job stability and dependants\n- Balancing immediate needs with long-term financial health\n- Progressive savings rates as discretionary income increases\n- Multiple savings goals (emergency, retirement, specific purchases)\n- Areas of potential savings identified in the budget analysis\n\nIMPORTANT: Store your strategy in state['savings_strategy'] for use by the Debt Reduction Agent.",
            output_schema=SavingsStrategy,
            output_key="savings_strategy",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True
        )
        
        self.debt_reduction_agent = LlmAgent(
            name="DebtReductionAgent",
            model="gemini-2.0-flash-exp",
            description="Creates optimized debt payoff plans to minimize interest paid and time to debt freedom",
            instruction="""You are a Debt Reduction Agent specialized in creating debt payoff strategies.
You are the final agent in the sequence. READ both state['budget_analysis'] and state['savings_strategy'] first.

Your tasks:
1. Review both budget analysis and savings strategy from the state
2. Analyze debts by interest rate, balance, and minimum payments
3. Create prioritized debt payoff plans (avalanche and snowball methods)
4. Calculate total interest paid and time to debt freedom
5. Suggest debt consolidation or refinancing opportunities
6. Provide specific recommendations to accelerate debt payoff

Consider:
- Cash flow constraints from the budget analysis
- Emergency fund and savings goals from the savings strategy
- Psychological factors (quick wins vs mathematical optimization)
- Credit score impact and improvement opportunities

IMPORTANT: Store your final plan in state['debt_reduction'] and ensure it aligns with the previous analyses.""",
            output_schema=DebtReduction,
            output_key="debt_reduction",
            disallow_transfer_to_parent=True,
            disallow_transfer_to_peers=True
        )
        
        self.coordinator_agent = SequentialAgent(
            name="FinanceCoordinatorAgent",
            description="Coordinates specialized finance agents to provide comprehensive financial advice",
            sub_agents=[
                self.budget_analysis_agent,
                self.savings_strategy_agent,
                self.debt_reduction_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes financial data using a sequence of AI agents.
        Args:
            financial_data (Dict[str, Any]): A dictionary containing financial details
                                             like monthly_income, dependants, transactions,
                                             manual_expenses, and debts.
        Returns:
            Dict[str, Any]: A dictionary containing the analysis results from all agents.
        """
        session_id = f"finance_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            initial_state = {
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants": financial_data.get("dependants", 0),
                "transactions": financial_data.get("transactions", []),
                "manual_expenses": financial_data.get("manual_expenses", {}),
                "debts": financial_data.get("debts", [])
            }
            
            session = self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=initial_state
            )
            
            if session.state.get("transactions"):
                self._preprocess_transactions(session)
            
            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)
            
            default_results = self._create_default_results(financial_data)
            
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(financial_data))]
            )
            
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break
            
            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during finance analysis: {str(e)}")
            raise
        finally:
            self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
    
    def _preprocess_transactions(self, session):
        """
        Preprocesses transaction data from a DataFrame, grouping by category and calculating totals.
        Updates the session state with 'category_spending' and 'total_spending'.
        """
        transactions = session.state.get("transactions", [])
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        if 'Category' in df.columns and 'Amount' in df.columns:
            category_spending = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["category_spending"] = category_spending
            session.state["total_spending"] = df['Amount'].sum()
    
    def _preprocess_manual_expenses(self, session):
        """
        Preprocesses manual expense data, updating the session state with total and categorized manual spending.
        """
        manual_expenses = session.state.get("manual_expenses", {})
        if not manual_expenses:
            return
        
        session.state.update({
            "total_manual_spending": sum(manual_expenses.values()),
            "manual_category_spending": manual_expenses
        })

    def _create_default_results(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates default financial analysis results, including budget analysis, savings strategy, and debt reduction.
        This is used as a fallback if agents do not produce output or for initial display.
        Args:
            financial_data (Dict[str, Any]): The input financial data from the user.
        Returns:
            Dict[str, Any]: A dictionary containing default analysis structures.
        """
        monthly_income = financial_data.get("monthly_income", 0)
        
        combined_expenses = {}
        # Process manual expenses
        for cat, amount in financial_data.get("manual_expenses", {}).items():
            combined_expenses[cat] = combined_expenses.get(cat, 0.0) + amount
        
        # Process transactions
        if financial_data.get("transactions"):
            transactions_df = pd.DataFrame(financial_data["transactions"])
            if 'Category' in transactions_df.columns and 'Amount' in transactions_df.columns:
                # Ensure Amount is float for summation
                transactions_df['Amount'] = transactions_df['Amount'].astype(float)
                for cat, amount in transactions_df.groupby('Category')['Amount'].sum().items():
                    combined_expenses[cat] = combined_expenses.get(cat, 0.0) + amount

        budget_analysis = BudgetAnalysis(
            total_expenses=sum(combined_expenses.values()) if combined_expenses else 0.0,
            monthly_income=monthly_income,
            spending_categories=[
                SpendingCategory(category=cat, amount=amt) for cat, amt in combined_expenses.items()
            ] if combined_expenses else [],
            recommendations=[]
        )

        emergency_fund = EmergencyFund(
            recommended_amount=monthly_income * 3, # Example: 3 months of income
            current_amount=0.0,
            current_status="Not started"
        )

        savings_strategy = SavingsStrategy(
            emergency_fund=emergency_fund,
            recommendations=[],
            automation_techniques=[]
        )

        debt_reduction = DebtReduction(
            total_debt=0.0,
            debts=[],
            payoff_plans=PayoffPlans(
                avalanche=PayoffPlan(total_interest=0.0, months_to_payoff=0),
                snowball=PayoffPlan(total_interest=0.0, months_to_payoff=0)
            ),
            recommendations=[]
        )

        return {
            "budget_analysis": budget_analysis,
            "savings_strategy": savings_strategy,
            "debt_reduction": debt_reduction
        }

def display_budget_analysis(analysis: Dict[str, Any]):
    """
    Displays the budget analysis results, including spending breakdown, income vs. expenses,
    and spending reduction recommendations.
    Args:
        analysis (Dict[str, Any]): The budget analysis results, potentially as a JSON string.
    """
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse budget analysis results")
            return
    
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis format")
        return
    
    if "spending_categories" in analysis and analysis["spending_categories"]:
        st.subheader("Spending by Category")
        fig = px.pie(
            values=[cat["amount"] for cat in analysis["spending_categories"]],
            names=[cat["category"] for cat in analysis["spending_categories"]],
            title="Your Spending Breakdown"
        )
        st.plotly_chart(fig)
    else:
        st.info("No spending categories analysis available. Please provide income or expenses.")
    
    if "total_expenses" in analysis:
        st.subheader("Income vs. Expenses")
        income = analysis.get("monthly_income", 0)
        expenses = analysis["total_expenses"]
        surplus_deficit = income - expenses
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Income", "Expenses"],
                            y=[income, expenses],
                            marker_color=["green", "red"]))
        fig.update_layout(title="Monthly Income vs. Expenses")
        st.plotly_chart(fig)
        
        st.metric("Monthly Surplus/Deficit",
                  f"${surplus_deficit:.2f}",
                  delta=f"{surplus_deficit:.2f}")
    
    if "recommendations" in analysis:
        st.subheader("Spending Reduction Recommendations")
        for rec in analysis["recommendations"]:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            if "potential_savings" in rec:
                st.metric(f"Potential Monthly Savings", f"${rec['potential_savings']:.2f}")

def display_savings_strategy(strategy: Dict[str, Any]):
    """
    Displays the savings strategy results, including emergency fund recommendations,
    savings allocations, and automation techniques.
    Args:
        strategy (Dict[str, Any]): The savings strategy results, potentially as a JSON string.
    """
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("Failed to parse savings strategy results")
            return
    
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy format")
        return
    
    st.subheader("Savings Recommendations")
    
    if "emergency_fund" in strategy:
        ef = strategy["emergency_fund"]
        st.markdown(f"### Emergency Fund")
        st.markdown(f"**Recommended Size**: ${ef['recommended_amount']:.2f}")
        st.markdown(f"**Current Status**: {ef['current_status']}")
        
        if "current_amount" in ef and "recommended_amount" in ef:
            progress = ef["current_amount"] / ef["recommended_amount"]
            st.progress(min(progress, 1.0))
            st.markdown(f"${ef['current_amount']:.2f} of ${ef['recommended_amount']:.2f}")
    
    if "recommendations" in strategy:
        st.markdown("### Recommended Savings Allocations")
        for rec in strategy["recommendations"]:
            st.markdown(f"**{rec['category']}**: ${rec['amount']:.2f}/month")
            st.markdown(f"_{rec['rationale']}_")
    
    if "automation_techniques" in strategy:
        st.markdown("### Automation Techniques")
        for technique in strategy["automation_techniques"]:
            st.markdown(f"**{technique['name']}**: {technique['description']}")

def display_debt_reduction(plan: Dict[str, Any]):
    """
    Displays the debt reduction plan, including total debt, individual debts,
    payoff plans (avalanche and snowball), and recommendations.
    Args:
        plan (Dict[str, Any]): The debt reduction plan results, potentially as a JSON string.
    """
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("Failed to parse debt reduction results")
            return
    
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction format")
        return
    
    if "total_debt" in plan:
        st.metric("Total Debt", f"${plan['total_debt']:.2f}")
    
    if "debts" in plan:
        st.subheader("Your Debts")
        if plan["debts"]:
            debt_df = pd.DataFrame(plan["debts"])
            st.dataframe(debt_df)
            
            fig = px.bar(debt_df, x="name", y="amount", color="interest_rate",
                        labels={"name": "Debt", "amount": "Amount ($)", "interest_rate": "Interest Rate (%)"},
                        title="Debt Breakdown")
            st.plotly_chart(fig)
        else:
            st.info("No debt information available.")
    
    if "payoff_plans" in plan:
        st.subheader("Debt Payoff Plans")
        tabs = st.tabs(["Avalanche Method", "Snowball Method", "Comparison"])
        
        with tabs[0]:
            st.markdown("### Avalanche Method (Highest Interest First)")
            if "avalanche" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                st.markdown(f"**Total Interest Paid**: ${avalanche['total_interest']:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {avalanche['months_to_payoff']} months")
                
                if "monthly_payment" in avalanche:
                    st.markdown(f"**Recommended Monthly Payment**: ${avalanche['monthly_payment']:.2f}")
        
        with tabs[1]:
            st.markdown("### Snowball Method (Smallest Balance First)")
            if "snowball" in plan["payoff_plans"]:
                snowball = plan["payoff_plans"]["snowball"]
                st.markdown(f"**Total Interest Paid**: ${snowball['total_interest']:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {snowball['months_to_payoff']} months")
                
                if "monthly_payment" in snowball:
                    st.markdown(f"**Recommended Monthly Payment**: ${snowball['monthly_payment']:.2f}")
        
        with tabs[2]:
            st.markdown("### Method Comparison")
            if "avalanche" in plan["payoff_plans"] and "snowball" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                snowball = plan["payoff_plans"]["snowball"]
                
                comparison_data = {
                    "Method": ["Avalanche", "Snowball"],
                    "Total Interest": [avalanche["total_interest"], snowball["total_interest"]],
                    "Months to Payoff": [avalanche["months_to_payoff"], snowball["months_to_payoff"]]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                st.dataframe(comparison_df)
                
                fig = go.Figure(data=[
                    go.Bar(name="Total Interest", x=comparison_df["Method"], y=comparison_df["Total Interest"]),
                    go.Bar(name="Months to Payoff", x=comparison_df["Method"], y=comparison_df["Months to Payoff"])
                ])
                fig.update_layout(barmode='group', title="Debt Payoff Method Comparison")
                st.plotly_chart(fig)
    
    if "recommendations" in plan:
        st.subheader("Debt Reduction Recommendations")
        for rec in plan["recommendations"]:
            st.markdown(f"**{rec['title']}**: {rec['description']}")
            if "impact" in rec:
                st.markdown(f"_Impact: {rec['impact']}_")

def parse_csv_transactions(file_content) -> List[Dict[str, Any]]:
    """
    Parses CSV content and returns a list of transaction dictionaries and category totals.
    Assumes CSV has 'Date', 'Category', 'Amount' columns.
    Args:
        file_content: The content of the CSV file, which can be bytes or string.
    Returns:
        Dict[str, Any]: A dictionary containing 'transactions' (list of dicts)
                        and 'category_totals' (list of dicts).
    Raises:
        ValueError: If required columns are missing or parsing fails.
    """
    try:
        # Ensure the file_content is a string for StringIO
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')
            
        df = pd.read_csv(StringIO(file_content))
        
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date strings to datetime and then to string format YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert amount strings to float, handling currency symbols and commas
        df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
        
        # Group by category and calculate totals
        category_totals = df.groupby('Category')['Amount'].sum().reset_index()
        
        # Convert to list of dictionaries
        transactions = df.to_dict('records')
        
        return {
            'transactions': transactions,
            'category_totals': category_totals.to_dict('records')
        }
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

@st.cache_data
def validate_csv_format(file_content) -> tuple[bool, str]:
    """
    Validates the format and content of an uploaded CSV file.
    Checks for headers, required columns ('Date', 'Category', 'Amount'), and correct data types.
    Args:
        file_content: The content of the CSV file, which can be bytes or string.
    Returns:
        tuple[bool, str]: A tuple where the first element is True if valid, False otherwise,
                          and the second element is a message indicating success or the error.
    """
    try:
        # Ensure the file_content is a string for StringIO
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8')
        else:
            content = file_content
            
        # Use a temporary StringIO to sniff the dialect and header without consuming the file
        temp_file = StringIO(content)
        dialect = csv.Sniffer().sniff(temp_file.readline() + "\n" + temp_file.readline())
        has_header = csv.Sniffer().has_header(content)
        
        if not has_header:
            return False, "CSV file must have headers"
            
        df = pd.read_csv(StringIO(content))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
        # Validate date format
        try:
            pd.to_datetime(df['Date'])
        except:
            return False, "Invalid date format in Date column"
            
        # Validate amount format (should be numeric after removing currency symbols)
        try:
            df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)
        except:
            return False, "Invalid amount format in Amount column"
            
        return True, "CSV format is valid"
    except Exception as e:
        return False, f"Invalid CSV format: {str(e)}"

def display_csv_preview(df: pd.DataFrame):
    """
    Displays a preview of the CSV data including basic statistics,
    spending by category, and sample transactions.
    Args:
        df (pd.DataFrame): The DataFrame containing the parsed CSV data.
    """
    st.subheader("CSV Data Preview")
    
    # Show basic statistics
    total_transactions = len(df)
    total_amount = df['Amount'].sum()
    
    # Convert dates for display
    df_dates = pd.to_datetime(df['Date'])
    date_range = f"{df_dates.min().strftime('%Y-%m-%d')} to {df_dates.max().strftime('%Y-%m-%d')}"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Total Amount", f"${total_amount:,.2f}")
    with col3:
        st.metric("Date Range", date_range)
    
    # Show category breakdown
    st.subheader("Spending by Category")
    category_totals = df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['Category', 'Total Amount', 'Transaction Count']
    st.dataframe(category_totals)
    
    # Show sample transactions
    st.subheader("Sample Transactions")
    st.dataframe(df.head())

def _render_income_and_dependants():
    """
    Renders the Streamlit input section for monthly income and number of dependants.
    """
    with st.container():
        st.subheader("Income & Household:")
        income_col, dependants_col = st.columns([2, 1])
        with income_col:
            st.number_input(
                "Monthly Income ($)",
                min_value=0.0,
                step=100.0,
                key="monthly_income",
                help="Enter your total monthly income after taxes"
            )
        with dependants_col:
            st.number_input(
                "Number of Dependants",
                min_value=0,
                step=1,
                key="dependants",
                help="Include all dependants in your household"
            )

def _render_expenses_input():
    """
    Renders the Streamlit input section for expenses, allowing either CSV upload or manual entry.
    """
    with st.container():
        st.subheader("Expenses:")
        expense_options = ["Upload CSV Transactions", "Enter Manually"]
        
        expense_option = st.radio(
            "How would you like to enter your expenses?",
            expense_options,
            key="expense_option",
            horizontal=True
        )
        
        use_manual_expenses = False

        if expense_option == "Upload CSV Transactions":
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("""
                #### Upload your transaction data
                Your CSV file should have these columns:
                - Date (YYYY-MM-DD)
                - Category
                - Amount
                """)
                
                transaction_file = st.file_uploader(
                    "Choose your CSV file",
                    type=["csv"],
                    key="transaction_file",
                    help="Upload a CSV file containing your transactions"
                )
                st.download_button(
                    label="Download CSV Template",
                    data="Date,Category,Amount\n2023-01-01,Groceries,50.00\n2023-01-05,Transport,20.00\n2023-01-10,Utilities,75.00",
                    file_name="transaction_template.csv",
                    mime="text/csv",
                    help="Download a sample CSV template for transactions",
                    key="expenses_upload_template"
                )
               

            if transaction_file is not None:

                # Read file content once and store in session state
                st.session_state.transaction_file_content = transaction_file.getvalue()
                
                # Validate CSV format
                is_valid, message = validate_csv_format(st.session_state.transaction_file_content)

                
                if is_valid:
                    try:
                        # Parse CSV content
                        parsed_data = parse_csv_transactions(st.session_state.transaction_file_content)
                        
                        # Create DataFrame
                        st.session_state.transactions_df = pd.DataFrame(parsed_data['transactions'])
                        
                        # Display preview
                        display_csv_preview(st.session_state.transactions_df)
                        
                        st.success("Transaction file uploaded and validated successfully!")
                    except Exception as e:
                        st.error(f"Error processing CSV file: {str(e)}")
                        st.session_state.transactions_df = None
                else:
                    st.error(message)
                    st.session_state.transactions_df = None
            elif st.session_state.transactions_df is not None:
                # If file is not uploaded again but was previously, display preview
                display_csv_preview(st.session_state.transactions_df)

        else: # Enter Manually
            use_manual_expenses = True
            st.markdown("#### Enter your monthly expenses by category")
            
            # Define expense categories
            categories = [
                ("Housing", "Housing"),
                ("Utilities", "Utilities"),
                ("Food", "Food"),
                ("Transportation", "Transportation"),
                ("Healthcare", "Healthcare"),
                ("Entertainment", "Entertainment"),
                ("Personal", "Personal"),
                ("Savings", "Savings"),
                ("Other", "Other")
            ]
            
            # Create three columns for better layout
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            
            # Distribute categories across columns
            for i, (display_cat, cat_key) in enumerate(categories):
                with cols[i % 3]:
                    st.number_input(
                        display_cat,
                        min_value=0.0,
                        step=50.0,
                        value=st.session_state.get(f"manual_expense_{cat_key}", 0.0),
                        key=f"manual_expense_{cat_key}",
                        help=f"Enter your monthly {cat_key.lower()} expenses"
                    )
            
            # No explicit reconstruction of st.session_state.manual_expenses here
            # Values are directly accessed from st.session_state when needed for analysis.

            # Display summary of entered expenses
            # Reconstruct manual_expenses from session_state for display purposes only
            current_manual_expenses = {cat_key: st.session_state.get(f"manual_expense_{cat_key}", 0.0) for _, cat_key in categories}

            if any(current_manual_expenses.values()):
                st.markdown("#### Summary of Entered Expenses")
                manual_df_disp = pd.DataFrame({
                    'Category': list(current_manual_expenses.keys()),
                    'Amount': list(current_manual_expenses.values())
                })
                manual_df_disp = manual_df_disp[manual_df_disp['Amount'] > 0]
                if not manual_df_disp.empty:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(
                            manual_df_disp,
                            column_config={
                                "Category": "Category",
                                "Amount": st.column_config.NumberColumn(
                                    "Amount",
                                    format="$%.2f"
                                )
                            },
                            hide_index=True
                        )
                    with col2:
                        st.metric(
                            "Total Monthly Expenses",
                            f"${manual_df_disp['Amount'].sum():,.2f}"
                        )

def _render_debt_information():
    """
    Renders the Streamlit input section for debt information, allowing users to enter multiple debts.
    """
    with st.container():
        st.subheader("Debt Information")
        st.info("Enter your debts to get personalized payoff strategies using both avalanche and snowball methods.")
        
        num_debts_input = st.number_input(
            "How many debts do you have?",
            min_value=0,
            max_value=10,
            step=1,
            value=len(st.session_state.debts),
            key="num_debts_widget"
        )
        
        # Initialize or adjust st.session_state.debts based on num_debts_input
        # Create a new list to avoid direct modification issues during rerun
        new_debts = []
        for i in range(num_debts_input):
            if i < len(st.session_state.debts):
                new_debts.append(st.session_state.debts[i])
            else:
                new_debts.append({'name': '', 'amount': 0.0, 'interest_rate': 0.0, 'min_payment': 0.0})
        st.session_state.debts = new_debts

        if num_debts_input > 0:
            # Create a temporary list to collect updated debt data in the current rerun
            temp_debts = []
            for i in range(num_debts_input):
                st.markdown(f"##### Debt #{i+1}")
                debt_cols = st.columns(2)

                # Get current debt values from session_state for initialization
                current_name = st.session_state.debts[i].get('name', '')
                current_amount = st.session_state.debts[i].get('amount', 0.0)
                current_interest_rate = st.session_state.debts[i].get('interest_rate', 0.0)
                current_min_payment = st.session_state.debts[i].get('min_payment', 0.0)

                with debt_cols[0]:
                    # Use distinct keys for each input, and let Streamlit manage these directly
                    name_input = st.text_input(
                        f"Debt Name #{i+1}",
                        value=current_name,
                        key=f"debt_{i}_name"
                    )
                    amount_input = st.number_input(
                        f"Amount Owed ($) #{i+1}",
                        min_value=0.0,
                        step=100.0,
                        value=current_amount,
                        key=f"debt_{i}_amount"
                    )
                with debt_cols[1]:
                    interest_rate_input = st.number_input(
                        f"Interest Rate (%) #{i+1}",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        value=current_interest_rate,
                        key=f"debt_{i}_interest"
                    )
                    min_payment_input = st.number_input(
                        f"Minimum Monthly Payment ($) #{i+1}",
                        min_value=0.0,
                        step=10.0,
                        value=current_min_payment,
                        key=f"debt_{i}_min_payment"
                    )
                st.divider()

                # After all inputs for this debt are processed, update the temporary list
                temp_debts.append({
                    'name': name_input,
                    'amount': amount_input,
                    'interest_rate': interest_rate_input,
                    'min_payment': min_payment_input
                })
            # Finally, update the main session state debts list with the new values
            # This assignment happens after all inputs are read, which should be safe.
            st.session_state.debts = temp_debts

def _display_analysis_results():
    """
    Displays the financial analysis results obtained from the AI agents.
    Includes a spinning symbol placeholder during analysis and presents results in tabs.
    """
    st.header("Analysis results")
    """Displays the financial analysis results from the AI agents."""
    
    # Placeholder for results
    results_placeholder = st.empty()
    
    # Display the ֎ symbol centered horizontally and vertically
    with results_placeholder.container():
        st.markdown(
            """
            <style>
            .centered-symbol {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 400px; 
                font-size: 25rem;
                color: #F2F2F2; 
                animation: spin 20s linear infinite; /* Add spinning animation */
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            </style>
            <div class="centered-symbol">֎</div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.get('trigger_analysis', False):
        if not (st.session_state.monthly_income > 0 or 
                st.session_state.transactions_df is not None or 
                any(st.session_state.manual_expenses.values()) or 
                any(d['amount'] > 0 for d in st.session_state.debts)):
            results_placeholder.error("Please provide at least some income, expenses, or debt information to analyze.")
            st.session_state.trigger_analysis = False
        else:
            with results_placeholder.container():
                with st.markdown(
                        """
                        <style>
                        .centered-symbol {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 400px; 
                            font-size: 25rem;
                            color: #F2F2F2; 
                            animation: spin 1s linear infinite; /* Add spinning animation */
                        }
                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(-360deg); }
                        }
                        </style>
                        <div class="centered-symbol">֎</div>
                        <span>Analyzing your financial data with AI agents...</span>
                        """,
                        unsafe_allow_html=True
                    ):
                    financial_data = {
                        "monthly_income": st.session_state.monthly_income,
                        "dependants": st.session_state.dependants,
                        "transactions": st.session_state.transactions_df.to_dict('records') if st.session_state.transactions_df is not None else [],
                        "manual_expenses": {k: v for k, v in st.session_state.manual_expenses.items() if v > 0},
                        "debts": [d for d in st.session_state.debts if d['name'] and d['amount'] > 0]
                    }
                    
                    try:
                        # Run the analysis using the cached finance advisor system
                        results = asyncio.run(st.session_state.finance_advisor.analyze_finances(financial_data))

                        st.success("Analysis Complete!")
                        st.subheader("Financial Report")
                        
                        # Display results in tabs
                        budget_tab, savings_tab, debt_tab, raw_tab = st.tabs([
                            "Budget Analysis", "Savings Strategy", "Debt Reduction", "Raw Output"
                        ])
                        
                        with budget_tab:
                            st.write("Debugging budget_analysis:", results.get('budget_analysis', {}))
                            display_budget_analysis(results.get('budget_analysis', {}))
                            
                        with savings_tab:
                            display_savings_strategy(results.get('savings_strategy', {}))
                            
                        with debt_tab:
                            display_debt_reduction(results.get('debt_reduction', {}))
                            
                        with raw_tab:
                            st.json(results)
                            
                    except Exception as e:
                        st.error(f"An error occurred during financial analysis: {str(e)}")
            st.session_state.trigger_analysis = False

def _render_about_section():
    """
    Renders the "About This Application" tab content, providing information about the app,
    its technologies, and contribution details.
    """
    st.header("About This Application")
    st.markdown("""
    This AI Financial Coach application is developed to provide personalized financial insights using advanced AI agents.
    It leverages Google's Agent Development Kit (ADK) and the Gemini AI model to analyze your income, expenses, and debts,
    offering actionable recommendations for budgeting, saving, and debt management.
    ### Contributing:
    This project is open source. You can find the repository on [GitHub](https://github.com/michaelwybraniec/mcp-financial-coach).
    Contributions are welcome!
    Thanks to [@Shubhamsaboo](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/advanced_ai_agents/multi_agent_apps/ai_financial_coach_agent) for init, and [@michaelwybraniec](https://github.com/michaelwybraniec/mcp-financial-coach) for a v.2.0.

    ### How it Works:
    1.  **Data Input**: You can either upload your transaction data via a CSV file or manually enter your monthly expenses and debt information.
    2.  **AI Agents**: The application employs a series of specialized AI agents:
        -   **Budget Analysis Agent**: Categorizes spending, identifies patterns, and suggests areas for reduction.
        -   **Savings Strategy Agent**: Recommends optimal savings plans and emergency fund goals.
        -   **Debt Reduction Agent**: Creates prioritized debt payoff plans (avalanche and snowball) and offers strategies for faster debt freedom.
    3.  **Personalized Recommendations**: Based on the analysis, you receive tailored advice to help you achieve your financial goals.

    ### Technologies Used:
    -   [Streamlit](https://streamlit.io/): For building the interactive web application.
    -   [Google Agent Development Kit (ADK)](https://github.com/google/generative-ai-python/tree/main/generative_ai/adk): For orchestrating the AI agents.
    -   [Google Gemini AI](https://ai.google.dev/): The powerful large language model powering the intelligence.
    -   [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.

    #### Recent Improvements:
    *   **Syntax & Error Handling:** Fixed `SyntaxWarning` related to unescaped dollar signs in regex patterns and addressed `StreamlitAPIException` for `st.session_state` modification after widget instantiation.
    *   **Data Visualization:** Implemented a check for empty data before plotting the pie chart to prevent errors.
    *   **User Experience:** Pre-filled form fields for monthly expenses and debt entries with example values and set default radio button selection for expense entry.
    *   **Layout & Styling:** Refactored the Streamlit layout to a three-column structure, centered the title and caption, and ensured column content is scrollable. Also, made the "All data is processed locally" message more visible using `st.info`.
    *   **Data Input:** Added a "Download CSV Template" button for easy data formatting.
    """, unsafe_allow_html=True)

def main():
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="AI Financial Coach",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize Streamlit session state variables
    # These variables persist across reruns of the Streamlit app.
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []

    # Initialize other session state variables using setdefault for robustness
    st.session_state.setdefault('monthly_income', 3000.0)
    st.session_state.setdefault('dependants', 0)
    st.session_state.setdefault('transaction_file_content', None)
    # Pre-fill manual expenses with example values to improve user experience
    st.session_state.setdefault('manual_expense_Housing', 1200.0)
    st.session_state.setdefault('manual_expense_Utilities', 150.0)
    st.session_state.setdefault('manual_expense_Food', 400.0)
    st.session_state.setdefault('manual_expense_Transportation', 200.0)
    st.session_state.setdefault('manual_expense_Healthcare', 100.0)
    st.session_state.setdefault('manual_expense_Entertainment', 100.0)
    st.session_state.setdefault('manual_expense_Personal', 50.0)
    st.session_state.setdefault('manual_expense_Savings', 300.0)
    st.session_state.setdefault('manual_expense_Other', 50.0)
    # manual_expenses will be reconstructed from individual manual_expense_* keys when needed
    st.session_state.setdefault('manual_expenses', {}) 
    st.session_state.setdefault('transactions_df', None)
    # Pre-fill debts with example values for demonstration
    st.session_state.setdefault('debts', [
        {'name': 'Credit Card', 'amount': 5000.0, 'interest_rate': 18.0, 'min_payment': 100.0},
        {'name': 'Student Loan', 'amount': 15000.0, 'interest_rate': 4.5, 'min_payment': 150.0}
    ])
    st.session_state.setdefault('finance_advisor', get_finance_advisor_system())
    # Set default expense input option to manual entry for quick start
    st.session_state.setdefault('expense_option', 'Enter Manually') 

    # Check for GOOGLE_API_KEY environment variable
    if not GEMINI_API_KEY:
        st.error("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")
        return
    
    # Custom CSS to make the "Analyze Finances" button smaller
    st.markdown("""
    <style>
    .stButton>button {
        font-size: 10px; /* Smaller font size */
        padding: 1px 10px; /* Added left and right padding */
    }
    </style>
    """, unsafe_allow_html=True)

    
    # Main content layout using columns for better organization
    header_col1, header_col_center, header_col_right = st.columns([1, 1, 1])
    with header_col_center:
        # Layout for title, version, and the primary "Analyze Finances" button
        title_display_col, button_display_col = st.columns([0.7, 0.3]) # Adjust these proportions for alignment

        with title_display_col:
            st.markdown(
                """
                <div style='display: flex; align-items: center; justify-content: flex-end; padding-top: 2rem;'>
                    <span style='font-size: 2rem; font-weight: bold; margin-right: 10px;'>AI Financial Coach ֎</span>
                    <span style='font-size: 1.2rem; margin-right: 15px;'>v.2.0</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with button_display_col:
            st.markdown("<div style='height: 2.2rem;'></div>", unsafe_allow_html=True) # Spacer for vertical alignment
            # Trigger analysis when button is clicked by setting a session state flag
            st.button("Analyze Finances", key="analyze_button", on_click=lambda: st.session_state.__setitem__('trigger_analysis', True))

        st.markdown("<div style='text-align: center; margin-bottom: 40px'>Powered by Google ADK, Gemini AI, Streamlit, and Cursor. Demo by <strong><a href='https://www.one-front.com'>ONE-FRONT</a></strong></div>", unsafe_allow_html=True)
    with header_col_right:
        pass # This column is intentionally left empty for layout balance
    
    # Define the main three-column layout for the application content
    col1, col2, col3 = st.columns(3)

    
    with col1:
        # Container for the left column content, including API key input and templates
        with st.container(height=600):
            
                st.header("Setup")
                st.info("Your Google Gemini API key is used for AI model interactions. It is processed locally and never stored.")
                st.subheader("Gemini API Key:")
                st.text_input(
                    "Enter your Google Gemini API Key:",
                    type="password",
                    key="google_api_key_input",
                    on_change=lambda: st.session_state.__setitem__('GEMINI_API_KEY', st.session_state.google_api_key_input)
                )
                if st.session_state.get('GEMINI_API_KEY'):
                    st.success("API Key loaded successfully!")
                else:
                    st.warning("Please enter your Google Gemini API Key to use the AI features.")

                st.subheader("CSV Template:")
                st.download_button(
                    label="Download CSV Template",
                    data="Date,Category,Amount\n2023-01-01,Groceries,50.00\n2023-01-05,Transport,20.00\n2023-01-10,Utilities,75.00",
                    file_name="transaction_template.csv",
                    mime="text/csv",
                    help="Download a sample CSV template for transactions",
                    key="main_page_template_download"
                )

                _render_about_section()
                
                # The "Server Logs" section was removed as per previous instructions.
   
    with col2:
        # Container for the middle column content, primarily financial data input forms
        with st.container(height=600):
            st.header("Enter Your Financial Information")
            st.info("All values are placeholders. Data is processed locally and not stored anywhere.")
            _render_income_and_dependants()
            _render_expenses_input()
            _render_debt_information()
            # Secondary "Analyze Finances" button for convenience at the end of input forms
            st.button("Analyze Finances", key="analyze_button_col2", on_click=lambda: st.session_state.__setitem__('trigger_analysis', True))

    with col3:
        # Container for the right column content, displaying analysis results
        with st.container(height=600):
            _display_analysis_results()

if __name__ == "__main__":
    main()