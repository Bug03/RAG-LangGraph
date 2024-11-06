from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == '__main__':
    print("Advanced CORRECTIVE RAG")
    # print(app.invoke(input({"question": "what is agents memory"})))
    # print(app.invoke(input={"question": "how to make pizza?"}))
