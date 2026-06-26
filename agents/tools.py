import json
import httpx
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from config import settings
from hr_policy.service import hr_vector_service  # Natively references the helper module cross-domain

@tool
async def search_hr_policies(query: str) -> str:
    """Search internal HR policies documents for rules, leaves guidelines, insurance benefits, structure, or procedures."""
    docs = hr_vector_service.search_policies(query)
    if not docs:
        return "No corresponding policy specifications found in current knowledge database files."
    return "\n---\n".join(docs)

@tool
async def search_employees_by_name(name: str, config: RunnableConfig) -> str:
    """Queries external corporate personnel system to search for an employee profile record by name match details."""
    configurable = config.get("configurable", {})
    token = configurable.get("auth_token", "")
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": token if token else f"Bearer {settings.EXTERNAL_DB_TOKEN}"}
        try:
            response = await client.get(
                f"{settings.EXTERNAL_DB_BASE_URL}/employee/search", 
                params={"name": name}, 
                headers=headers,
                timeout=5.0
            )
            return json.dumps(response.json(), indent=2) if response.status_code == 200 else f"Failed: {response.status_code}"
        except Exception as e:
            return f"Error executing communication pipeline connection: {str(e)}"

@tool
async def get_employee_by_id(employee_id: int, config: RunnableConfig) -> str:
    """Fetches comprehensive employee details including role, status, joining date, experience, address, and departments via their ID."""
    configurable = config.get("configurable", {})
    token = configurable.get("auth_token", "")
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": token if token else f"Bearer {settings.EXTERNAL_DB_TOKEN}"}
        try:
            response = await client.get(
                f"{settings.EXTERNAL_DB_BASE_URL}/employee/{employee_id}", 
                headers=headers, 
                timeout=5.0
            )
            return json.dumps(response.json(), indent=2) if response.status_code == 200 else f"Failed: {response.status_code}"
        except Exception as e:
            return f"Database interface connection failure: {str(e)}"

@tool
async def list_departments(config: RunnableConfig) -> str:
    """Lists every official department registered inside company master database tables."""
    configurable = config.get("configurable", {})
    token = configurable.get("auth_token", "")
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": token if token else f"Bearer {settings.EXTERNAL_DB_TOKEN}"}
        try:
            response = await client.get(
                f"{settings.EXTERNAL_DB_BASE_URL}/department", 
                headers=headers, 
                timeout=5.0
            )
            return json.dumps(response.json(), indent=2) if response.status_code == 200 else f"Failed: {response.status_code}"
        except Exception as e:
            return f"Network layer error: {str(e)}"

tools_list = [search_hr_policies, search_employees_by_name, get_employee_by_id, list_departments]