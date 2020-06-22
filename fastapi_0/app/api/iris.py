from fastapi import APIRouter
from . import _iris


router = APIRouter()


@router.get("/test")
def test():
    return _iris.test()
