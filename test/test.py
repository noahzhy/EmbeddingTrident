import requests
from fastapi import FastAPI
from ray import serve
import ray

ray.init()   # ⭐加这一行

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:

    @app.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"


serve.run(FastAPIDeployment.bind(), route_prefix="/")

print(
    requests.get(
        "http://localhost:8000/hello",
        params={"name": "Theodore"}
    ).json()
)