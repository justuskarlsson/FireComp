"""
Lazy-loading data source registry.

All data sources are instantiated lazily on first access to avoid
opening files that may not exist (e.g., on desktop without cluster data).
"""

from functools import cached_property


class _LazyDSRC:
    """Registry of data sources with lazy initialization."""

    @cached_property
    def fires_daily(self):
        from firecomp.dsrc.vnp14_dsrc import FiresDaily

        return FiresDaily()

    @cached_property
    def fires_next_day(self):
        from firecomp.dsrc.vnp14_dsrc import FiresDailyNextDay

        return FiresDailyNextDay()

    @cached_property
    def fires_accum(self):
        from firecomp.dsrc.vnp14_dsrc import FiresAccum

        return FiresAccum()

    @cached_property
    def ae_embeddings(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings()

    @cached_property
    def ae_embeddings_64(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(band_list=list(range(1, 65)))

    @cached_property
    def ae_y17_64(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(year=2017, band_list=list(range(1, 65)))

    @cached_property
    def ae_y18_64(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        # TODO: Change to year=2018 when data is downloaded
        return AEEmbeddings(year=2018, band_list=list(range(1, 65)))

    @cached_property
    def ae_y19_64(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        # TODO: Change to year=2019 when data is downloaded
        return AEEmbeddings(year=2019, band_list=list(range(1, 65)))

    @cached_property
    def ae_y20_64(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(year=2020, band_list=list(range(1, 65)))

    @cached_property
    def ae_y21_64(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(year=2021, band_list=list(range(1, 65)))

    @cached_property
    def ae_embeddings_12345(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(band_list=[1, 2, 3, 4, 5])

    @cached_property
    def ae_embeddings_pca_5(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(pca_bands=5)

    @cached_property
    def ae_y20_pca_5(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(year=2020, pca_bands=5)

    @cached_property
    def ae_y21_pca_5(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(year=2021, pca_bands=5)

    @cached_property
    def ae_y21_12345(self):
        from firecomp.dsrc.ae_embedding import AEEmbeddings

        return AEEmbeddings(year=2021, band_list=[1, 2, 3, 4, 5])

    @cached_property
    def era5(self):
        from firecomp.dsrc.weather import ERA5

        return ERA5()

    @cached_property
    def era5_fire(self):
        from firecomp.dsrc.weather import ERA5Fire

        return ERA5Fire()  # Ratio-based anomaly (no pre-computation needed)

    @cached_property
    def era5_fire_pct(self):
        from firecomp.dsrc.weather import ERA5FirePercentile

        return ERA5FirePercentile()  # Percentile-based (needs climatology first)

    @cached_property
    def weathernext2_forecast(self):
        from firecomp.dsrc.weathernext2 import WeatherNext2Forecast

        return WeatherNext2Forecast()

    @cached_property
    def canopy_height(self):
        from firecomp.dsrc.canopy_height import CanopyHeightMeta

        return CanopyHeightMeta()

    @cached_property
    def canopy_height_date(self):
        from firecomp.dsrc.canopy_height import CanopyHeightDate

        return CanopyHeightDate()

    @cached_property
    def gfs_forecast(self):
        from firecomp.dsrc.gfs import GFSForecast

        return GFSForecast(forecast_days=7)

    @cached_property
    def gfs_1day(self):
        """GFS 1-day forecast for next_day task (5 channels)."""
        from firecomp.dsrc.gfs import GFSForecast

        return GFSForecast(forecast_days=1)

    @cached_property
    def gfs_7day(self):
        """GFS 7-day forecast (35 channels)."""
        from firecomp.dsrc.gfs import GFSForecast

        return GFSForecast(forecast_days=7)

    @cached_property
    def weather_forecast(self):
        # Alias for gfs_forecast
        return self.gfs_forecast


# Singleton instance
dsrc = _LazyDSRC()
