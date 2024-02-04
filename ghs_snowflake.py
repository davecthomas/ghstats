from typing import Dict, Any, List
import os
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas


def get_db_env() -> Dict[str, str]:
    dict_db_env: Dict[str, str] = {}
    dict_db_env["snowflake_user"] = os.getenv("SNOWFLAKE_USER")
    dict_db_env["snowflake_password"] = os.getenv("SNOWFLAKE_PASSWORD")
    dict_db_env["snowflake_account"] = os.getenv("SNOWFLAKE_ACCOUNT")
    dict_db_env["snowflake_warehouse"] = os.getenv("SNOWFLAKE_WAREHOUSE")
    dict_db_env["snowflake_db"] = os.getenv("SNOWFLAKE_DB")
    dict_db_env["snowflake_schema"] = os.getenv("SNOWFLAKE_SCHEMA")
    return dict_db_env


def get_snowflake_connection():
    dict_db_env: Dict[str, str] = get_db_env()
    conn = snowflake.connector.connect(
        user=dict_db_env["snowflake_user"],
        password=dict_db_env["snowflake_password"],
        account=dict_db_env["snowflake_account"],
        warehouse=dict_db_env["snowflake_warehouse"],
        database=dict_db_env["snowflake_db"],
        schema=dict_db_env["snowflake_schema"]
    )
    return conn
