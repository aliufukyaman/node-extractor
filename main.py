import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from action_extractor import router


def main():
    app = FastAPI()
    app.include_router(router, prefix="/api")

    # swagger
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title="Action Extractor",
            version="0.0.1",
            description="Action extractor API Documentation",
            routes=app.routes,
        )
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    @app.get("/", include_in_schema=False)
    async def docs_redirect():
        return RedirectResponse(url='/docs')

    uvicorn.run(app, host="0.0.0.0", port=8081)


if __name__ == "__main__":
    main()
