"""
AWS Bedrock abstraction with deterministic mock fallback.
All prompts use the Anthropic Messages API format via InvokeModel.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from config.settings import Settings, get_settings
from utils.logger import get_logger


class BedrockService:
    """
    AWS Bedrock abstraction. Falls back to deterministic mock responses
    if MOCK_BEDROCK=True or credentials are unavailable.
    """

    MODEL_ID = "us.anthropic.claude-sonnet-4-6"

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or get_settings()
        self.mock_mode = self.config.MOCK_BEDROCK
        self._client = None
        self.logger = get_logger(__name__)

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                kwargs = {"region_name": self.config.AWS_REGION}
                if self.config.AWS_ACCESS_KEY_ID:
                    kwargs["aws_access_key_id"] = self.config.AWS_ACCESS_KEY_ID
                if self.config.AWS_SECRET_ACCESS_KEY:
                    kwargs["aws_secret_access_key"] = self.config.AWS_SECRET_ACCESS_KEY
                self._client = boto3.client("bedrock-runtime", **kwargs)
            except Exception as e:
                self.logger.warning("Bedrock client init failed, switching to mock", error=str(e))
                self.mock_mode = True
        return self._client

    def _invoke(
        self, system_prompt: str, user_message: str, max_tokens: int = 1024
    ) -> str:
        """Core invocation with retry logic."""
        if self.mock_mode:
            return self._mock_response("_invoke", user_message=user_message)

        client = self._get_client()
        if self.mock_mode:
            return self._mock_response("_invoke", user_message=user_message)

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        })

        for attempt in range(3):
            try:
                response = client.invoke_model(
                    modelId=self.config.AWS_BEDROCK_MODEL_ID,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                return result["content"][0]["text"]
            except Exception as e:
                err_str = str(e)
                if "ThrottlingException" in err_str and attempt < 2:
                    wait = 2 ** (attempt + 1)
                    self.logger.warning(f"Throttled, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    self.logger.error("Bedrock invocation failed", error=err_str)
                    return self._mock_response("_invoke", user_message=user_message)

        return self._mock_response("_invoke", user_message=user_message)

    def executive_summary(self, portfolio_summary: dict, top_recs: list) -> str:
        """Generate a 3-paragraph executive summary for the buying team."""
        if self.mock_mode:
            return self._mock_response("executive_summary")

        system = (
            "You are an expert eyewear retail buying strategist. "
            "Write concise, data-driven executive summaries for buying teams."
        )
        user = (
            f"Portfolio Summary:\n{json.dumps(portfolio_summary, indent=2)}\n\n"
            f"Top 10 SKU Recommendations:\n{json.dumps(top_recs[:10], indent=2)}\n\n"
            "Write a 3-paragraph executive summary:\n"
            "Para 1: Immediate risks requiring action this week.\n"
            "Para 2: Top 3 revenue opportunities.\n"
            "Para 3: Strategic recommendation (trend, supplier, category).\n"
            "Be specific with numbers. Keep it under 300 words."
        )
        return self._invoke(system, user, max_tokens=512)

    def explain_sku_recommendation(self, sku_data: dict) -> str:
        """Plain-English explanation under 120 words for a busy buyer."""
        if self.mock_mode:
            return self._mock_response("explain_sku_recommendation", sku_data=sku_data)

        system = (
            "You are an expert eyewear buying assistant. "
            "Explain SKU recommendations in plain English. Be concise, no jargon."
        )
        user = (
            f"SKU Data:\n{json.dumps(sku_data, indent=2)}\n\n"
            "Write 2-3 punchy sentences (under 120 words) covering: "
            "demand trend, social signal, supply risk, and margin impact. "
            "Write as if briefing a busy retail buyer."
        )
        return self._invoke(system, user, max_tokens=200)

    def answer_buying_question(
        self, question: str, context: dict, history: list[dict]
    ) -> str:
        """Conversational Q&A with multi-turn context."""
        if self.mock_mode:
            return self._mock_response("answer_buying_question", question=question)

        system = (
            "You are an expert AI eyewear buying assistant with deep knowledge of retail, "
            "inventory management, demand forecasting, and supply chain optimization. "
            f"Current business context:\n{json.dumps(context, indent=2)}\n\n"
            "Answer questions accurately using the provided data. Be concise and actionable."
        )
        # Keep last 10 turns
        trimmed_history = history[-10:]
        messages = trimmed_history + [{"role": "user", "content": question}]

        client = self._get_client()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "system": system,
            "messages": messages,
        })

        for attempt in range(3):
            try:
                response = client.invoke_model(
                    modelId=self.config.AWS_BEDROCK_MODEL_ID,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(response["body"].read())
                return result["content"][0]["text"]
            except Exception as e:
                if "ThrottlingException" in str(e) and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    return self._mock_response("answer_buying_question", question=question)

        return self._mock_response("answer_buying_question", question=question)

    def generate_trend_report(self, trending_skus: list, social_data: dict) -> str:
        """Weekly trend report in structured markdown."""
        if self.mock_mode:
            return self._mock_response("generate_trend_report", trending_skus=trending_skus)

        system = (
            "You are a fashion-forward eyewear trend analyst. "
            "Generate insightful weekly trend reports for buying teams."
        )
        user = (
            f"Trending SKUs (top 15):\n{json.dumps(trending_skus[:15], indent=2)}\n\n"
            f"Social Signal Summary:\n{json.dumps(social_data, indent=2)}\n\n"
            "Generate a structured markdown trend report with:\n"
            "# Weekly Trend Report\n"
            "## Rising Stars (top 3 SKUs with momentum)\n"
            "## Fading Styles (declining SKUs to watch)\n"
            "## Platform Insights (which social platforms are driving demand)\n"
            "## Recommended Actions (3-5 specific buying actions)\n"
            "Keep total length under 400 words."
        )
        return self._invoke(system, user, max_tokens=600)

    # ── Mock responses ────────────────────────────────────────────────────────
    def _mock_response(self, method_name: str, **kwargs) -> str:
        """Returns realistic-looking mock text per method."""
        mocks = {
            "executive_summary": (
                "**Immediate Risks:** This week, 12 SKUs are critically low on inventory with less than "
                "7 days of supply remaining. SKU0042 (Premium Titanium Aviator) is our highest-urgency item "
                "with a projected stockout in 3 days and a £18,500 revenue-at-risk exposure. Three key "
                "suppliers in the high-risk band require immediate reorder placement to avoid holiday season gaps.\n\n"
                "**Top Revenue Opportunities:** (1) Premium Acetate Cat-Eye frames show 34% velocity "
                "acceleration driven by TikTok trendsetters — projecting £42K in incremental margin if "
                "restocked within 14 days. (2) Women's luxury titanium collection is trending 2.8x above "
                "seasonal norm across Instagram. (3) Online channel growth of 23% YoY is concentrated in "
                "the 25-34 age segment — premium SKUs in this range warrant deeper inventory positions.\n\n"
                "**Strategic Recommendation:** Reduce acetate dependency on Italian suppliers (currently at "
                "medium geopolitical concentration risk) by qualifying one alternative EU supplier before Q3. "
                "Accelerate clearance of 8 overstock SKUs carrying £31K in holding costs through a targeted "
                "15–25% markdown campaign to free capital for the trending premium titanium category."
            ),
            "explain_sku_recommendation": (
                "This SKU is accelerating fast — sales are up 38% over the past 30 days, driven by "
                "strong Instagram traction with 1,200+ organic mentions this week. Stock will run out in "
                "under 10 days at current velocity. With a 52% gross margin and a reliable supplier "
                "delivering in 45 days, ordering now secures an estimated £8,200 profit opportunity."
            ),
            "answer_buying_question": (
                f"Based on the current data, {kwargs.get('question', 'your question')} can be addressed as follows: "
                "The top 3 SKUs requiring immediate restocking are SKU0042, SKU0087, and SKU0153, all with "
                "less than 7 days of supply and accelerating demand. I recommend placing reorders with your "
                "primary suppliers this week to avoid stockouts ahead of the upcoming seasonal peak. "
                "Would you like me to break down the order quantities and estimated costs for each?"
            ),
            "generate_trend_report": (
                "# Weekly Trend Report — Week of 2024-12-09\n\n"
                "## 🌟 Rising Stars\n"
                "- **SKU0042 Premium Titanium Aviator**: +340% social mentions on TikTok, sentiment 0.91. Reorder urgently.\n"
                "- **SKU0117 Acetate Cat-Eye**: Instagram reach up 5x, influencer-driven. High conversion expected.\n"
                "- **SKU0203 Oversized Round Sunglasses**: Pinterest momentum building — ideal for SS24 push.\n\n"
                "## 📉 Fading Styles\n"
                "- **SKU0015 Rectangle TR90**: Velocity declining 28% MoM. Review markdown strategy.\n"
                "- **SKU0091 Budget Metal Wayfarer**: Return rate elevated at 22%. Pause reordering.\n\n"
                "## 📱 Platform Insights\n"
                "TikTok is driving the strongest sales conversion for premium frames (18-34 segment). "
                "Pinterest is emerging as primary discovery channel for luxury sunglasses.\n\n"
                "## ✅ Recommended Actions\n"
                "1. Increase SKU0042 buy depth by 300 units.\n"
                "2. Allocate 20% of budget to Cat-Eye and Oversized silhouettes.\n"
                "3. Mark down 8 stagnant Rectangle SKUs by 20% to free capital.\n"
                "4. Brief top 3 Instagram influencers on new Acetate collection launch.\n"
                "5. Monitor SKU0091 defect rate — escalate with supplier if return rate persists."
            ),
            "_invoke": (
                "This is a mock AI response. Set MOCK_BEDROCK=false and configure AWS credentials "
                "to enable real Bedrock responses."
            ),
        }
        return mocks.get(method_name, "Mock response not available for this method.")


if __name__ == "__main__":
    cfg = get_settings()
    svc = BedrockService(cfg)

    print("=== Executive Summary (mock) ===")
    summary = svc.executive_summary(
        {"critical_skus": 5, "total_buy_budget_usd": 120000},
        [{"sku_id": "SKU0042", "urgency_score": 0.95}]
    )
    print(summary)

    print("\n=== SKU Explanation (mock) ===")
    print(svc.explain_sku_recommendation({"sku_id": "SKU0042", "urgency_score": 0.95}))
