# Onder Nul

**Track:** Finance
**Team:** Vayun Godara

A two-day warning for the hours when Dutch electricity prices go below zero ("onder nul"). Code and results: [onder_nul.ipynb](onder_nul.ipynb). Demo: download this repository and open `demo/index.html`.

## The problem

The Dutch hourly electricity price was below zero for 584 hours in 2025. About 573,000 households have a contract that follows that price, and from 2027 solar owners can no longer cancel out the power they export against the power they use. In a negative hour, exporting solar power costs money and charging a car is cheapest. Tomorrow's prices only come out around 13:00 the day before, too late to plan a weekend.

## What I built

A model that gives every hour a chance of a negative price, two days ahead. In go the calendar, past prices and the weather forecast as it looked two days before. Out comes a probability per hour, and a warning at 0.4 or higher. On the demo page you pick a date in 2026, read the advice, then reveal what really happened.

## What I added

The idea is my own. The model only gets weather forecasts that existed at the time, taken from an archive of old forecasts. I also tested how many days ahead the warning stays useful and whether following it would save money.

## How it works

Data: 24,072 hourly day-ahead prices for the Netherlands, February 2024 to 13 November 2026 (energy-charts.info, none missing), and archived forecasts of sun, wind and temperature for De Bilt, the North Sea and central Germany (Open-Meteo). I dropped the first five weeks, which have no 7-day forecasts. I split by date and never shuffled: train to June 2025, choose settings on July to December 2025, test on 2026 (7,607 hours, 404 negative).

Model: gradient boosted decision trees from scikit-learn, because the data is a table and trees learn combinations such as "sunny and a weekend" by themselves. A leak check scrambles later prices and confirms the model's inputs do not change.

## Results

A model that never warns is 95% accurate and useless. So I report precision (warnings that were right) and recall (negative hours caught) on 2026, two days ahead.

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Same hour last week was negative | 0.38 | 0.38 | 0.38 |
| Weekend and forecast sun above 350 W/m² | 0.47 | 0.54 | 0.50 |
| One small decision tree | 0.31 | 0.83 | 0.45 |
| My model | 0.65 | 0.67 | 0.66 |

When I redraw the test days at random 1,000 times, the model's F1 stays between 0.59 and 0.71 and beats the best rule every time. Seven days ahead its F1 is still 0.52. The money is small: stopping solar export in the warned hours gains about €10 over the test period, out of a possible €16.

Where it fails: it caught none of the 20 negative hours at night, which come from wind. On weekdays 49% of its warnings were right, against 74% at weekends. Its worst miss was Easter Sunday, with 3 of 17 hours caught, and public holidays are not an input. There is only one test year.

This project is a prototype and is not financial advice.

## What I'd do next

1. Add public holidays as an input and check whether the holiday misses shrink.
2. Repeat the test on 2027 and on another country's prices.

## Sources

- Data: [energy-charts.info API](https://api.energy-charts.info) (Fraunhofer ISE, data from Bundesnetzagentur | SMARD.de, CC BY 4.0) and the [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) and Forecast API (CC BY 4.0).
- Background figures: [Rijksoverheid on the end of the salderingsregeling](https://www.rijksoverheid.nl/themas/klimaat-milieu-en-natuur/energie-thuis/salderingsregeling) and [Solar Magazine reporting ACM figures](http://solarmagazine.nl/nieuws-zonne-energie/i43038/573-000-nederlanders-hebben-dynamisch-energiecontract-marktaandeel-anwb-krimpt).
- Libraries: pandas, NumPy, scikit-learn, Matplotlib, Requests, Jupytext. No pretrained models and no tutorial code. Typeface: Barlow (SIL Open Font Licence).
- AI tools: [Claude](https://claude.ai) (Anthropic), through the [Amp](https://ampcode.com) coding agent. It helped me compare ideas, wrote most of the code and the demo page, made the screen recording for the video and drafted this write-up. Inside Amp it also used [Parallel](https://parallel.ai) web search for research and the [TypeSafe Jev](https://docs.typesafe.ai) model to check this write-up's numbers against the notebook. I picked the project, checked the results and edited the text.

To run it: `pip install -r requirements.txt jupyter`, then open `onder_nul.ipynb`. It reads the saved files in `data/` and rewrites `demo/data.js`, the only thing the demo page reads.
