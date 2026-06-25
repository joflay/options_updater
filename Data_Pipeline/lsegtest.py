import lseg.data as ld

APP_KEY = '14470904ee5d45b28b4f34b9f227beafbee75476'
ld.open_session(app_key=APP_KEY)

df = ld.get_history(
    universe="MSFT.O",
    fields=["TR.PriceClose", "TR.PriceOpen", "TR.PriceHigh", "TR.PriceLow"]
)

print(df.head())

ld.close_session()