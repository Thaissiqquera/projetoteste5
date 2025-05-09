from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
import sys
import os

# Adicionar o diretório pai ao path para poder importar o módulo main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar a aplicação FastAPI principal
from main import app as main_app

# Reexportar a aplicação para o Vercel
app = main_app
