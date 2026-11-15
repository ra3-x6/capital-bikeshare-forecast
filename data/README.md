# Data provenance

Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894).

`day.csv` is the unmodified daily file from the [official UCI download](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip). It contains observed daily Capital Bikeshare rentals in Washington, D.C., from January 2011 through December 2012. The data is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The forecast reads only `dteday` and `cnt`. Its predictors come from prior observed rental counts and the calendar. Same-day weather and the component counts `casual` and `registered` are excluded from the predictors.
