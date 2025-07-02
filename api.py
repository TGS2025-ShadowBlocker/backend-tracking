from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"kick": True, "punch": True}

if __name__ == "__main__":
    from uvicorn import run
    run(app)