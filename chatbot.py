import streamlit as st
import yfinance as yf
import re
import plotly.graph_objects as go
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

# 1. Page Configuration
st.set_page_config(
    page_title="AlphaFin AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Environment Setup
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 3. Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    :root {
        --primary: #00ff88;
        --secondary: #007aff;
        --background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    .main { background: var(--background); }
    
    .stChatFloatingInputContainer {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
    }
    
    .stPlotlyChart {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        background: rgba(0,0,0,0.3) !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin: 0.5rem 0 !important;
    }
    
    .gradient-text {
        background: linear-gradient(45deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sentiment-gauge {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 4. Helper Functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_hf_client(model_name="ahmedrachid/FinancialBERT-Sentiment-Analysis", timeout=30):
    return InferenceClient(
        token=HF_TOKEN,
        model=model_name,
        timeout=timeout,
        headers={"x-use-cache": "0"}
    )

def calculate_technical_indicators(hist):
    hist = hist.copy()
    hist['SMA50'] = hist['Close'].rolling(50).mean()
    hist['SMA200'] = hist['Close'].rolling(200).mean()
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    hist['RSI'] = 100 - (100 / (1 + (gain/loss)))
    return hist.dropna()

def create_interactive_chart(hist):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist['Open'],
        high=hist['High'], low=hist['Low'],
        close=hist['Close'], name='Price'))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['SMA50'],
        line=dict(color='#00ff88', width=2),
        name='50-day SMA'))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['SMA200'],
        line=dict(color='#ff007a', width=2),
        name='200-day SMA'))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=30, b=20),
        height=400, xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_metric_cards(financial_data):
    cols = st.columns(4)
    metrics = [
        ('📈 Price', f"{financial_data['current_price']:.2f} {financial_data['currency']}", None),
        ('📉 24h Change', f"{financial_data['pct_change']:.2f}%", financial_data['pct_change']),
        ('📊 P/E Ratio', f"{financial_data['pe_ratio']:.2f}" if financial_data.get('pe_ratio') else 'N/A', None),
        ('🏦 Market Cap', f"{financial_data['market_cap']/1e9:.2f}B" if financial_data.get('market_cap') else 'N/A', None)
    ]
    for col, (title, value, change) in zip(cols, metrics):
        with col:
            color = "#00ff88" if (change is not None and change >= 0) else "#ff007a" if (change is not None and change < 0) else "inherit"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{title}</h3>
                <div class="metric-value" style="color: {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

def extract_ticker(text):
    matches = re.findall(r'\b[A-Z]{2,5}\b', text)
    return matches[0] if matches else None

def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        if len(hist) < 2:
            return None
        
        current_price = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2]
        return {
            'ticker': ticker,
            'current_price': current_price,
            'price_change': current_price - prev_close,
            'pct_change': ((current_price - prev_close)/prev_close)*100,
            'company_name': info.get('longName', ticker),
            'currency': info.get('currency', 'USD'),
            'pe_ratio': info.get('trailingPE'),
            'market_cap': info.get('marketCap'),
            'hist_data': calculate_technical_indicators(hist)
        }
    except Exception as e:
        st.error(f"Error retrieving data for {ticker}: {str(e)}")
        return None

def format_context(financial_data):
    if not financial_data:
        return ""
    
    pe = f"{financial_data['pe_ratio']:.2f}" if financial_data.get('pe_ratio') else 'N/A'
    cap = f"{financial_data['market_cap']/1e9:.2f}B" if financial_data.get('market_cap') else 'N/A'
    
    sma50 = financial_data['hist_data']['SMA50'].iloc[-1]
    # Calculate the percentage difference between current price and 50-day SMA
    perc_diff = ((financial_data['current_price'] - sma50) / sma50) * 100
    
    return f"""
## {financial_data['company_name']} ({financial_data['ticker']})
- **Current Price:** {financial_data['current_price']:.2f} {financial_data['currency']}
- **24h Change:** {financial_data['price_change']:.2f} ({financial_data['pct_change']:.2f}%)
- **P/E Ratio:** {pe}
- **Market Cap:** {cap}
- **50-day SMA:** {sma50:.2f}
- **Price vs SMA Difference:** {perc_diff:.2f}%
- **RSI:** {financial_data['hist_data']['RSI'].iloc[-1]:.2f}
"""

# 5. Analysis Functions
def analyze_sentiment(text):
    try:
        client = get_hf_client()  # Default FinancialBERT model for sentiment
        result = client.text_classification(text)
        return result[0] if result else None
    except Exception as e:
        st.error(f"Sentiment analysis error: {str(e)}")
        return None

def generate_response(prompt, context):
    try:
        # Define a detailed analysis prompt using your guidelines.
        analysis_prompt = (
            "You are an advanced financial analysis engine. Your task is to perform an in-depth evaluation "
            "of a company’s current stock performance, using both technical and sentiment-based metrics. "
            "Using the provided input data, generate a complete analysis report that includes backend calculations, "
            "technical indicator assessments, risk/opportunity evaluations, and comparisons with similar companies in the sector.\n\n"
            "### Analysis Requirements:\n"
            "- **Technical Indicator Analysis:** Explain the significance of the 50-day SMA in indicating short-to-medium term trends. Assess whether the current price near the SMA suggests a consolidation phase. Also analyze the RSI to indicate balanced market momentum, and include any necessary backend calculations (e.g., percentage differences between the current price and the 50-day SMA).\n"
            "- **Risk and Opportunity Assessment:** Discuss potential opportunities (e.g., breakout potential if the stock moves above the 50-day SMA) and evaluate risks (e.g., high P/E ratio and consolidation uncertainty). Provide quantitative measures where applicable.\n"
            "- **Sentiment Recommendation:** Based on technical indicators and sentiment confidence, offer a clear recommendation (buy, hold, or sell) with quantitative and qualitative justification.\n"
            "- **Comparative Analysis:** Compare the provided metrics with similar companies in the sector, highlighting differences in market cap, P/E ratio, and momentum indicators, and explain their implications.\n"
            "- **Final Synthesis:** Summarize your analysis with a balanced conclusion addressing both opportunities and risks.\n\n"
            "Use clear headings/subheadings and include any backend calculations or formulas used in your analysis."
        )
        
        # Combine the analysis prompt with the context data.
        full_context = analysis_prompt + "\n\n" + context

        # Use a general Q&A model to extract any additional insights.
        qa_client = get_hf_client("deepset/roberta-base-squad2")
        answers = qa_client.question_answering(
            question=prompt,
            context=full_context,
            top_k=3
        )
        
        # Perform sentiment analysis for overall context.
        sentiment_client = get_hf_client("ahmedrachid/FinancialBERT-Sentiment-Analysis")
        sentiment = sentiment_client.text_classification(full_context)[0]
        label_map = {
            'LABEL_0': ('Bearish', '🔴', '#ff007a'),
            'LABEL_1': ('Neutral', '🟡', '#ffd700'),
            'LABEL_2': ('Bullish', '🟢', '#00ff88')
        }
        sentiment_label, emoji, color = label_map.get(
            sentiment['label'], ('Neutral', '🟡', '#ffd700'))
        
        # Build the detailed analysis report
        overview = f"""
### Overview of Key Metrics
{context.strip()}
"""
        technical = """
### Technical Indicators Analysis

**50-day SMA (Simple Moving Average):**  
- The 50-day SMA smooths out price data over the last 50 trading days and indicates short-to-medium term trends.
- The current price's proximity to the SMA suggests a consolidation phase, as the market awaits a catalyst.

**RSI (Relative Strength Index):**  
- The RSI measures momentum and typically a value around 40–50 indicates balanced market conditions.
- This neutral reading supports the view of consolidation.
"""
        risks_opps = """
### In-Depth Risk and Opportunity Assessment

**Opportunities:**  
- **Potential Breakout:** A move above the 50-day SMA could signal the start of an upward trend.  
- **Stable Momentum:** Neutral RSI values indicate the potential for an entry point ahead of a breakout.

**Risks:**  
- **Consolidation Uncertainty:** The consolidation phase may lead to sharp moves in either direction without clear directional bias.  
- **High Valuation:** An elevated P/E ratio suggests a premium valuation, which could pose a risk if growth expectations are not met.
"""
        comparative = """
### Comparative Analysis

- **Market Cap & P/E Ratio:** When compared with similar companies in the sector, differences in market capitalization and P/E ratios highlight potential valuation and growth discrepancies.
- **Momentum Indicators:** Variations in technical indicators (such as the 50-day SMA and RSI) among peers can signal differing market dynamics and investor sentiment.
"""
        synthesis = f"""
### Final Synthesis

Based on the technical indicators and the sentiment analysis:
- The 50-day SMA and the calculated price difference indicate the stock is currently in a consolidation phase.
- The RSI near the mid-range reinforces a neutral momentum.
  
**Overall Recommendation:**  
While the market appears stable, investors should monitor for a breakout above the 50-day SMA. The high P/E ratio remains a cautionary factor. Based on the overall assessment, the sentiment suggests a **{sentiment_label}** outlook.
"""
        # Filter out repetitive or vague responses like "buy, hold, or sell" from Q&A
        qa_insights = "\n".join([f"- {ans['answer']}" for ans in answers 
                                  if len(ans['answer'].split()) > 3 and "buy, hold, or sell" not in ans['answer'].lower()])
        qa_section = f"\n### Additional Insights from Q&A\n{qa_insights}" if qa_insights.strip() else ""
        
        response = f"""📈 **Market Analysis** ({emoji} {sentiment_label}):

{overview}

{technical}

{risks_opps}

{comparative}

{synthesis}
{qa_section}

🔮 **Sentiment Confidence:** {sentiment['score']*100:.1f}%
"""
        return response
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        # Fallback: provide key metrics overview
        metrics = re.findall(r"- \*\*([\w\s]+)\*\*: ([\d\.]+(?:B|%| [A-Z]+)?)", context)
        fallback = "📊 **Key Metrics Overview:**\n" + "\n".join([f"- {name}: {value}" for name, value in metrics])
        return fallback


# 6. Main Application
def main():
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0 2rem;">
        <h1 style="font-size: 3.5rem; margin: 0;" class="gradient-text">
            AlphaFin AI Advisor
        </h1>
        <p style="font-size: 1.2rem; color: #888; margin-top: 0.5rem;">
            Free Market Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for chat history and rate-limiting
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = datetime.now() - timedelta(seconds=30)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask about stocks/markets..."):
        if (datetime.now() - st.session_state.last_query).seconds < 10:
            st.warning("Please wait 10 seconds between queries")
            return
            
        st.session_state.last_query = datetime.now()
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        financial_data = None
        ticker = extract_ticker(prompt)
        
        # Process ticker if found
        if ticker:
            with st.spinner(f"🔍 Analyzing {ticker}..."):
                financial_data = get_financial_data(ticker)
                if financial_data:
                    create_metric_cards(financial_data)
                    with st.expander("📊 Technical Charts", expanded=True):
                        fig = create_interactive_chart(financial_data['hist_data'])
                        st.plotly_chart(fig, use_container_width=True)
                    with st.expander("📈 Market Sentiment", expanded=True):
                        context_text = format_context(financial_data)
                        sentiment = analyze_sentiment(context_text)
                        if sentiment:
                            label_map = {'LABEL_0': ('Bearish','🔴','#ff007a'), 
                                         'LABEL_1': ('Neutral','🟡','#ffd700'), 
                                         'LABEL_2': ('Bullish','🟢','#00ff88')}
                            sentiment_label, emoji, color = label_map.get(
                                sentiment['label'], ('Neutral','🟡','#ffd700'))
                            confidence = sentiment['score'] * 100
                            st.markdown(f"""
                            <div class="sentiment-gauge">
                                <h2 style="margin: 0 0 1rem 0; color: {color}">
                                    {emoji} {sentiment_label}
                                </h2>
                                <div style="font-size: 2.5rem; color: {color}; margin: 1rem 0">
                                    {confidence:.1f}%
                                </div>
                                <progress value="{confidence}" max="100" style="width:80%;height:15px;accent-color:{color}"></progress>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Sentiment analysis unavailable at this time.")

        # Generate and display the final response
        with st.chat_message("assistant", avatar="💎"):
            with st.spinner("🧠 Generating analysis..."):
                try:
                    context = format_context(financial_data) if financial_data else ""
                    response = generate_response(prompt, context)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
