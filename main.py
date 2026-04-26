from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}