import os
import traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import constants as ct
import utils

# ==============================================================
# 初期化処理
# ==============================================================
def initialize():
    """アプリ全体の初期化処理"""
    print("✅ initialize() started")

    try:
        # -----------------------------
        # 1. 環境変数ロード
        # -----------------------------
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI APIキーが設定されていません。")

        print("🔑 OpenAI APIキーを取得しました")

        # -----------------------------
        # 2. データディレクトリ準備
        # -----------------------------
        persist_directory = os.path.join("logs", "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        print(f"📂 ChromaDB用ディレクトリ: {persist_directory}")

        # -----------------------------
        # 3. Embedding モデル初期化
        # -----------------------------
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("🧠 Embeddingsモデルを初期化しました")

        # -----------------------------
        # 4. ドキュメント読み込み
        # -----------------------------
        docs_dir = os.path.join("logs", "docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)

        # 対応するファイル形式をロード
        loaders = []
        if os.listdir(docs_dir):
            loaders.append(DirectoryLoader(docs_dir, glob="*.txt", loader_cls=TextLoader))
            loaders.append(DirectoryLoader(docs_dir, glob="*.pdf", loader_cls=PyPDFLoader))
            loaders.append(DirectoryLoader(docs_dir, glob="*.docx", loader_cls=Docx2txtLoader))

            documents = []
            for loader in loaders:
                try:
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ 一部のドキュメントで読み込み失敗: {e}")

            print(f"📄 {len(documents)} 件のドキュメントをロードしました")
        else:
            documents = []
            print("⚠️ ドキュメントフォルダが空です。空の状態で続行します。")

        # -----------------------------
        # 5. テキスト分割
        # -----------------------------
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            print(f"🧩 分割後ドキュメント数: {len(texts)}")
        else:
            texts = []

        # -----------------------------
        # 6. ベクトルDB作成
        # -----------------------------
        if texts:
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vectordb.persist()
            print("💾 ChromaDBへベクトルを登録しました")
        else:
            print("⚠️ ベクトル化するドキュメントがありません。ChromaDBの生成をスキップします。")

        print("✅ initialize() completed")

    except Exception as e:
        print(f"❌ initialize() failed: {e}")
        traceback.print_exc()
        raise e
