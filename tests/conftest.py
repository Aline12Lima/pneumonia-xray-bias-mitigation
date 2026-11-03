"""Configuração dos testes: sem marca de conferência de hashes por sessão (cada teste confere tudo)."""
import os

os.environ["MARCAS_HASHES"] = ""
