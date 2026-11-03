import os
import streamlit as st
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# 페이지 기본 설정
st.set_page_config(page_title="Gemini + Qdrant RAG 챗봇", page_icon="🤖", layout="centered")

st.title("📚 내 문서 기반 Qdrant & Gemini RAG 챗봇")
st.markdown("Qdrant Cloud와 Gemini API를 연동하여, 업로드한 문서를 기반으로 정확하고 신뢰할 수 있는 답변을 제공하는 챗봇입니다.")

# 사이드바 설정 (API 키 입력)
st.sidebar.header("🔑 API 및 클라우드 설정")
gemini_api_key = st.sidebar.text_input("GEMINI_API_KEY", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
qdrant_url = st.sidebar.text_input("QDRANT_URL", value=os.environ.get("QDRANT_URL", ""))
qdrant_api_key = st.sidebar.text_input("QDRANT_API_KEY", type="password", value=os.environ.get("QDRANT_API_KEY", ""))

collection_name = "my_rag_documents"

if not gemini_api_key or not qdrant_url or not qdrant_api_key:
    st.warning("⚠️ 사이드바에 GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY를 모두 입력해주세요.")
else:
    # 1. Gemini 클라이언트 초기화
    client_genai = genai.Client(api_key=gemini_api_key)

    # 2. Qdrant 클라이언트 및 임베딩 모델 초기화 (컬렉션 자동 생성 포함)
    @st.cache_resource
    def init_resources(url, api_key):
        q_client = QdrantClient(url=url, api_key=api_key)
        embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 컬렉션이 없으면 자동으로 생성 (404 에러 방지)
        if not q_client.collection_exists(collection_name):
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # bge-small-en-v1.5 모델의 벡터 차원 수
                    distance=models.Distance.COSINE
                )
            )
        return q_client, embed_model

    try:
        q_client, embed_model = init_resources(qdrant_url, qdrant_api_key)
    except Exception as e:
        st.error(f"Qdrant 연결 오류: {e}")

    # 사이드바: 문서 데이터 추가 섹션
    st.sidebar.subheader("📄 문서 데이터 추가")
    user_document = st.sidebar.text_area("학습시킬 텍스트 내용을 입력하세요:", height=150)
    
    if st.sidebar.button("문서 벡터화 및 저장"):
        if user_document.strip():
            with st.spinner("문서를 벡터화하여 Qdrant Cloud에 저장하는 중..."):
                q_client.add(
                    collection_name=collection_name,
                    documents=[user_document],
                    metadata=[{"source": "streamlit_input"}]
                )
            st.sidebar.success("✅ 문서가 Qdrant에 성공적으로 저장되었습니다!")
        else:
            st.sidebar.error("저장할 텍스트를 입력해주세요.")

    # 3. 챗봇 대화 화면 구성
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("문서 내용에 대해 무엇이든 물어보세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Qdrant에서 관련 문서를 검색하고 Gemini가 답변을 생성하는 중..."):
                try:
                    # 사용자 질문을 벡터로 변환
                    query_vector = list(embed_model.embed([prompt]))[0].tolist()

                    # Qdrant에서 벡터 기반 검색 수행
                    response_points = q_client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        limit=3
                    )

                    search_results = response_points.points
                    context_text = "\n\n".join([hit.payload.get("document", "") for hit in search_results if hit.payload and "document" in hit.payload]) if search_results else ""
                    
                    if not context_text and search_results:
                        context_text = "\n\n".join([hit.document for hit in search_results if hasattr(hit, 'document') and hit.document]) if search_results else "관련된 문서를 찾지 못했습니다."

                    if not context_text.strip():
                        context_text = "관련된 문서를 찾지 못했습니다."

                    # Gemini 프롬프트 작성
                    full_prompt = f"""다음 참고 문서 내용을 바탕으로 사용자의 질문에 정확하게 답변해 주세요. 문서에 없는 내용은 모른다고 답하세요.

[참고 문서]
{context_text}

[사용자 질문]
{prompt}
"""

                    # Gemini API 호출
                    response = client_genai.interactions.create(
                        model="Gemini-3.5-Flash-Lite",
                        input=full_prompt
                    )
                    answer = response.output_text

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_message = f"오류가 발생했습니다: {e}"
                    st.error(error_message)
