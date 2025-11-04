import os
import traceback
from typing import List

# LangChain core modules
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableSequence

# OpenAI / LLM
from langchain_openai import ChatOpenAI

# その他共通
import constants as ct


# ==============================================================
# ContextualCompressionRetriever（削除された機能の代替実装）
# ==============================================================
class ContextualCompressionRetriever(BaseRetriever):
    """LangChain 0.3+ 互換: コンテキスト圧縮リトリーバー代替"""

    def __init__(self, base_retriever: BaseRetriever):
        self.base_retriever = base_retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        """同期で関連文書を取得"""
        docs = self.base_retriever.get_relevant_documents(query)
        # 圧縮処理（今回は簡略化）
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """非同期で関連文書を取得"""
        docs = await self.base_retriever.aget_relevant_documents(query)
        return docs


# ==============================================================
# チャットモデル初期化
# ==============================================================
def init_chat_model(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """OpenAIモデル初期化"""
    try:
        return ChatOpenAI(
            model=model_name,
            temperature=0.3,
            streaming=False
        )
    except Exception as e:
        print(f"❌ モデル初期化失敗: {e}")
        traceback.print_exc()
        raise


# ==============================================================
# Streamlit エラー出力用
# ==============================================================
def build_error_message(message: str) -> str:
    """Streamlit 用の統一エラーメッセージ"""
    return f"❌ エラーが発生しました: {message}"


# ==============================================================
# 検索・回答生成用パイプライン（必要に応じて）
# ==============================================================
def create_basic_retriever_chain(retriever: VectorStoreRetriever, llm: ChatOpenAI) -> RunnableSequence:
    """Retriever + LLM の基本パイプラインを構築"""
    try:
        chain = RunnableSequence(
            steps=[
                ("retriever", retriever),
                ("llm", llm)
            ]
        )
        return chain
    except Exception as e:
        print(f"❌ RetrieverChain 初期化エラー: {e}")
        traceback.print_exc()
        raise


# ==============================================================
# デバッグ用メッセージ
# ==============================================================
def log_debug_info():
    """デバッグ用に主要情報を出力"""
    print("🔧 utils.py loaded successfully")
    print(f"🔧 OpenAI API Key: {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
# ============================================================
# Chat応答生成用の関数
# ============================================================
def generate_answer(prompt: str, mode: str):
    """
    入力されたプロンプトとモードに基づいて応答を生成する関数。
    社内文書検索 or 社内問い合わせモードを切り替えて処理。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        if mode == "社内文書検索":
            system_prompt = (
                "あなたは社内ドキュメント検索アシスタントです。"
                "入力に関連する社内文書の内容を要約し、正確かつ簡潔に説明してください。"
            )
        else:
            system_prompt = (
                "あなたは社内問い合わせ対応アシスタントです。"
                "質問の背景を考慮し、利用者が知りたい情報を文脈から推測して答えてください。"
            )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])

        chain = prompt_template | llm
        result = chain.invoke({"input": prompt})
        return result.content

    except Exception as e:
        print(f"❌ generate_answer() failed: {e}")
        return f"エラーが発生しました: {e}"
