from openai import OpenAI
import chromadb
from app.core.config import settings


# ========= 1) Fireworks Client =========

client = OpenAI(
    api_key=settings.FIREWORKS_API_KEY,
    base_url="https://api.fireworks.ai/inference/v1"
)

# أسماء النماذج من الإعدادات
EMBED_MODEL = settings.EMBEDDING_MODEL
LLM_MODEL = settings.LLM_MODEL


def get_embeddings(text_list):
    """إرجاع Embeddings من Qwen3-Embedding-8B (بُعد 4096)."""
    clean = [t.replace("\n", " ") for t in text_list]
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=clean,
    )
    return [d.embedding for d in resp.data]


# ========= 2) ChromaDB Connection =========

db = chromadb.PersistentClient(path=settings.CHROMA_PATH)
collection = db.get_collection("grammar_collection")


# ========= 3) Context Retrieval =========

def retrieve_context(question: str, top_k: int = 3):
    """استرجاع أفضل k مقاطع من قاعدة البيانات."""
    embedding = get_embeddings([question])

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents"]
    )

    docs = results.get("documents", [[]])[0]

    if not docs:
        return None
    
    return "\n---\n".join(docs)


# ========= 4) System Instruction =========

SYSTEM_INSTRUCTION = """
أنت مستشار لغوي متخصص في النحو والصرف والبلاغة العربية.

## قواعد صارمة يجب الالتزام بها:

1. **لا تقم بتصحيح النصوص أبداً:**
   - إذا طلب منك المستخدم تصحيح جملة أو نص أو فقرة، ارفض بأدب وقل:
     "عذراً، أنا متخصص في الإجابة عن الأسئلة النحوية والصرفية والبلاغية فقط، ولا أقوم بتصحيح النصوص."
   - لا تقم بتصحيح أخطاء إملائية أو نحوية في نصوص يرسلها المستخدم.
   - لا تعد صياغة أي نص حتى لو طُلب منك ذلك.

2. **أجب فقط عن الأسئلة النحوية والصرفية والبلاغية:**
   - أسئلة الإعراب، القواعد النحوية، الأوزان الصرفية، الأساليب البلاغية، وما شابهها.
   - إذا كان السؤال خارج هذا النطاق (مثل الترجمة، التلخيص، كتابة المقالات، أو أي موضوع آخر)، اعتذر بأدب.

3. **اعتمد على السياق المعطى (RAG) كمرجع أساسي أولاً:**
   - إذا كان السياق يحتوي على معلومات مفيدة، اعتمد عليه بشكل رئيسي.
   - إذا لم يغطِّ السياق السؤال بالكامل، يمكنك الاستعانة بمعرفتك العامة بشكل مختصر وبسيط.

4. **جاوب بالعربية فقط، بأسلوب علمي واضح ومختصر.**
"""


# ========= 5) RAG Answer Generation =========

def rag_answer(question: str) -> str:
    """يعطي جواب باستخدام RAG أولاً، وإذا لم يجد يستعين بالمعرفة العامة."""
    
    context = retrieve_context(question)

    # بناء البرومبت بحسب وجود سياق RAG أو عدمه
    if context:
        prompt = f"""
السياق المسترجع من قاعدة البيانات:
{context}

سؤال المستخدم:
{question}
"""
    else:
        # لم يُعثر على سياق في RAG → يستعين بالمعرفة العامة بشكل بسيط
        prompt = f"""
لم يتوفر سياق من قاعدة البيانات لهذا السؤال.
أجب من معرفتك العامة في النحو والصرف والبلاغة بشكل مختصر وبسيط، مع الالتزام بجميع القواعد.

سؤال المستخدم:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    return response.choices[0].message.content
