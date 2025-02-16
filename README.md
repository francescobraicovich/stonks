# stonks
Trading based on sentiment analysis on reddit.


reddit-finance-pipeline/
├── config/
│   └── config.yml
├── data_ingestion/
│   ├── __init__.py
│   ├── pushshift_reddit.py
│   ├── yahoo_finance.py
│   └── data_cleaner.py
├── data_storage/
│   ├── __init__.py
│   └── database_manager.py
├── pipelines/
│   ├── __init__.py
│   └── wsb_price_pipeline.py
├── main.py
├── requirements.txt
└── README.md
