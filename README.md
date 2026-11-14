# Onder Nul

**Track:** Finance
**Team:** Vayun Godara

A two-day warning for the hours when Dutch electricity prices go below zero ("onder nul").

## The problem

In the Netherlands the hourly electricity price was negative for 584 hours in 2025. About 573,000 households have a contract that follows that price, and from 1 January 2027 solar owners can no longer cancel out the power they export against the power they use. In a negative hour, exporting solar power costs money and charging a car is cheapest. Tomorrow's prices only come out around 13:00 the day before, which is too late to plan a weekend.

## What I built

A model that gives every hour of a day a chance of the price going below zero, two days before that day. In go the calendar, the prices up to two days before, and the weather forecast as it looked two days before. Out comes a probability per hour, and a warning when it is 0.4 or higher.

The demo page (`demo/index.html`) shows any day of 2026: the warning from two days earlier next to the prices that really happened, with every hit, miss and false alarm marked. A household would open it on Thursday to plan Saturday.

## What I added

Our own idea. It is not one of the suggested projects.

The model only gets weather forecasts that existed at the time, taken from an archive of old forecasts, because nobody knows the real weather two days ahead. I also asked two questions of my own: how many days ahead the warning stays useful, and whether following it would have saved money.

## How it works

Data: hourly day-ahead prices for the Netherlands from energy-charts.info, February 2024 to 13 November 2026 (24,072 hours, none missing; 15-minute prices averaged to hours). Archived forecasts of sunshine, wind and temperature come from the Open-Meteo Previous Runs API for De Bilt, a point in the North Sea and central Germany. I dropped the first five weeks, where the 7-day forecasts are missing.

I split by date and never shuffled: training to June 2025 (11,209 hours), validation July to December 2025 (4,416 hours), test 1 January to 13 November 2026 (7,607 hours, 404 of them negative).

Model: gradient boosted decision trees (scikit-learn's `HistGradientBoostingClassifier`). The data is a table, and trees can learn combinations such as "sunny and a weekend" without me writing them in. I chose the inputs and the warning threshold on the validation months, then retrained on everything before 2026. A leak check in the notebook scrambles all prices after one date and confirms the inputs for the next 48 hours do not change.

## Results

A model that never warns is 95% accurate and useless, so I report precision (how often a warning was right) and recall (how many negative hours were caught), on the 2026 test data, two days ahead.

| Method | Precision | Recall |
|---|---|---|
| Never warn | 0 | 0 |
| Same hour last week was negative | 0.38 | 0.38 |
| Weekend and forecast sun above 350 W/m² | 0.47 | 0.54 |
| My model | 0.65 | 0.67 |

Each group of inputs helped: F1 went from 0.48 with only the calendar and past prices to 0.57 with the Dutch forecast, 0.62 with North Sea wind and 0.66 with German weather. One day ahead the F1 is 0.66 and seven days ahead 0.52, while the last-week rule stays at 0.38.

The money is small. A home with ten solar panels that stopped exporting in the warned hours would have gained about €10 over the test period, out of a possible €16; the last-week rule would have lost about €8. Choosing which of the next three days to charge a car (40 kWh) would have cost €2.20 per charge with the model, €2.73 with the sunniest forecast day and €2.95 by always charging tomorrow.

Where it fails:

- It caught none of the 20 negative hours in the evening or at night. Those come from wind, and almost all the negative hours it trained on came from midday sun.
- On weekdays 49% of its warnings were right, against 74% at weekends.
- It caught 90% of the hours below −20 euros per MWh but 51% of the hours less than 1 euro below zero.
- Its worst miss was Easter Sunday, 5 April 2026: 17 negative hours, 3 caught. Public holidays are not an input.
- Its worst false alarm was 3 May 2026, when the forecast promised about twice the sunshine that arrived.

There is only one test year, so I can't say how well this holds in 2027.

This project is a prototype and is not financial advice.

## What I'd do next

1. Add public holidays and a forecast of German wind and solar production as inputs, then check whether the night and holiday misses shrink.
2. Run the same test on German prices, where there are more years of data, to see if the result holds outside one country and one year.

## Sources

- Prices: [energy-charts.info API](https://api.energy-charts.info) by Fraunhofer ISE, data from Bundesnetzagentur | SMARD.de, CC BY 4.0.
- Weather forecasts: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) and Forecast API, CC BY 4.0.
- Background figures: [Rijksoverheid on the end of the salderingsregeling](https://www.rijksoverheid.nl/themas/klimaat-milieu-en-natuur/energie-thuis/salderingsregeling), and [Solar Magazine reporting ACM figures](http://solarmagazine.nl/nieuws-zonne-energie/i43038/573-000-nederlanders-hebben-dynamisch-energiecontract-marktaandeel-anwb-krimpt) for the 573,000 dynamic contracts on 1 January 2026.
- Libraries: pandas, NumPy, scikit-learn, Matplotlib, Requests, Jupytext. No pretrained models and no tutorial code.
- Typeface on the demo page: Barlow by Jeremy Tribby, SIL Open Font Licence.
- AI tools: Claude (Anthropic), used through the [Amp](https://ampcode.com) coding agent. It helped me compare project ideas, wrote most of the code and the demo page, and drafted this write-up. I picked the project, checked the results and edited the text.

## Run it

```
pip install -r requirements.txt jupyter
jupyter notebook onder_nul.ipynb
```

The notebook reads the saved files in `data/`. If they are missing, it downloads them again. Running it rewrites `demo/data.js`, which is the only thing the demo page reads. Open `demo/index.html` in a browser to see the demo.

| File | What it is |
|---|---|
| `onder_nul.ipynb` | The whole project with outputs and a plain-English note on every step |
| `onder_nul.py` | The same notebook as a script, kept in sync with Jupytext |
| `data/` | The prices and archived weather forecasts the results were computed from |
| `demo/` | The demo page, its data file and its fonts |
