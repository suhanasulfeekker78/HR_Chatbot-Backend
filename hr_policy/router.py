from typing import List
from fastapi import APIRouter, UploadFile, File, status
from fastapi.responses import JSONResponse
from hr_policy.service import hr_vector_service

router = APIRouter(prefix="/hr", tags=["HR Document Manager"])

@router.get("/documents")
async def list_hr_documents():
    """Returns a clean list of all filenames currently processed and saved in the vector database."""
    try:
        filenames = hr_vector_service.get_all_document_names()
        return {"status": "success", "documents": filenames}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "detail": str(e)}
        )

@router.post("/documents")
async def upload_hr_documents(files: List[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            content = await file.read()
            hr_vector_service.add_document(filename=file.filename, file_bytes=content)
            uploaded_files.append(file.filename)
        return {"status": "success", "message": f"Successfully ingested: {uploaded_files}"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "detail": str(e)}
        )

@router.delete("/documents/{filename}")
async def remove_hr_document(filename: str):
    try:
        hr_vector_service.delete_document(filename)
        return {"status": "success", "message": f"Removed policy tracks for '{filename}'"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "detail": str(e)}
        )