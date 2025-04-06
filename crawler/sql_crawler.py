import mysql.connector
import logging

class SqlHandler:
    def __init__(self, config):
        try:
            self.db = mysql.connector.connect(
                host=config.get("sql", {}).get("HOST"),
                user=config.get("sql", {}).get("USER"),
                password=config.get("sql", {}).get("PASSWORD"),
                database=config.get("sql", {}).get("DATABASE"),
            )
            self.cursor = self.db.cursor()
        except:
            logging.error("SQL Bridge connection failed")