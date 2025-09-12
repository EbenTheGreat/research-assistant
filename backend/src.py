from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.upload_pdfs import router as upload_pdfs
from backend.routes.ask_questions import router as ask_question
from backend.middlewares.exception_handlers import catch_exception_middleware

version = "v1"
app = FastAPI(title="AI-Powered RAG Guitar Assistant",
              description="An Api that uses RAG to store and retrieve the history of the guitar",
              version=version,
              license_info={
                  "name": "MIT",
                  "url": "https://opensource.org/licenses/mit"
              },
              contact={
                  "name": "Eben-The-Great",
                  "email": "tijaniebenezer6@gmail.com",
                  "url": "https://github.com/EbenTheGreat/BookApp"
              })

@app.get('/')
async def read_root():
    return {"message": "welcome"}


# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# middleware exception handlers
app.middleware("http")(catch_exception_middleware)

# routers

# Upload pdf documents
app.include_router(upload_pdfs)
# asking query
app.include_router(ask_question)




