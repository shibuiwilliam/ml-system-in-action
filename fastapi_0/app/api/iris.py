from fastapi import APIRouter
from . import _iris
import logging

logger = logging.getLogger(__name__) 
router = APIRouter()


@router.get("/test")
def test():
    return _iris.test()
