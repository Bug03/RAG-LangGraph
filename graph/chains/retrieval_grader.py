from langchain_core.prompts import ChatPromptTemplate

# Pydantic giúp việc xử lý và kiểm tra dữ liệu đầu vào trở nên dễ dàng
# bằng cách sử dụng Python Classes để định nghĩa cấu trúc dữ liệu
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_openai import ChatOpenAI

## temperature=0: Điều chỉnh nhiệt độ của mô hình LLM.
## Nhiệt độ bằng 0 khiến cho mô hình trả về các kết quả có tính quyết định cao hơn (ít ngẫu nhiên hơn).
llm = ChatOpenAI(temperature=0)


class GradeDocuments(BaseModel):
    """ Binary score for relevance check on retrieve documents """

    # recieve a string Yes or No
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)
"""
This line initializes a structured LLM (Language Learning Model) grader.

- `llm`: An instance of `ChatOpenAI` with a specified temperature.
- `GradedDocuments`: A Pydantic model class that defines the structure of the output, 
  which includes a binary score indicating the relevance of retrieved documents.

The `with_structured_output` method configures the LLM to produce outputs that conform 
to the structure defined by the `GradedDocuments` class ( Yes or No ).
"""

system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keywords(s) or semantic meaning related to the question,, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User questions: {question}"),
    ]
)

## retrieval_grader: Đây là bộ kết hợp giữa lời nhắc (grade_prompt) và mô hình ngôn ngữ có đầu ra có cấu trúc (structured_llm_grader).
# Phép toán | được sử dụng để kết hợp hai phần này lại với nhau,
# tạo thành một công cụ hoàn chỉnh để đánh giá mức độ liên quan của tài liệu dựa trên câu hỏi của người dùng.
# Kết quả sẽ được trả về dưới dạng nhị phân "yes" hoặc "no
retrieval_grader = grade_prompt | structured_llm_grader
