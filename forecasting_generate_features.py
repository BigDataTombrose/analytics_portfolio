# Generate lag features for the current year (e.g., week -1, -2, etc.)
def lag_features_this_year(new_signals, metric, date_field, dimension_list, agg_list):
    lagged_agg_columns = []

    for week_number in [1, 2, 3, 4]:  # Use 1–4 week lags
        col_prefix = f'week_neg_{week_number}_'
        num_days = week_number * 7  # Convert weeks to days

        for agg_type in agg_list:
            col_name = col_prefix + agg_type
            if len(dimension_list) > 0:
                # Grouped rolling window aggregation per dimension
                group_df = new_signals.groupby(by=dimension_list)[metric]
                reset_index_list = list(range(0, len(dimension_list)))
                new_signals[col_name] = group_df.rolling(
                    window=num_days, closed='left', min_periods=num_days
                ).agg(agg_type).reset_index(level=reset_index_list, drop=True)
            else:
                # Global rolling aggregation
                group_df = new_signals[metric]
                new_signals[col_name] = group_df.rolling(
                    window=num_days, closed='left', min_periods=num_days
                ).agg(agg_type)

            lagged_agg_columns.append(col_name)

    return new_signals, lagged_agg_columns


# Returns the ISO week number of the last week of the previous year
def lastweeknumberoflastyear(date_input):
    from datetime import datetime
    return datetime(int(date_input.year - 1), 12, 28).isocalendar()[1]


# Generate lag features from N years ago
def lag_features_x_years_ago(new_signals, metric, date_field, feature_list, dimension_list, lagged_agg_columns, years_ago_int):
    lag_values = new_signals[[date_field, metric] + dimension_list + lagged_agg_columns]

    col_prefix = f"year_neg_{years_ago_int}_"
    
    # Rename columns to avoid collisions and clarify temporal lag
    for col in lag_values.columns:
        if col not in dimension_list:
            lag_values = lag_values.rename(columns={col: col_prefix + col})
    
    # Remove non-feature columns
    column_names_no_date = lag_values.columns.tolist()
    for col_name in dimension_list:        
        column_names_no_date.remove(col_name)
    for col_name in [date_field]:
        column_names_no_date.remove(col_prefix + date_field)

    # Ensure numeric type and collect new feature names
    for col in column_names_no_date:
        lag_values[col] = lag_values[col].astype(float)
        feature_list.append(col)

    return lag_values, feature_list


# Combine lag features across the last 2 years + year-over-year signals
def lag_features(new_signals, metric, date_field, feature_list, dimension_list, lagged_agg_columns, agg_list):
    from datetime import timedelta
    import numpy as np
    
    # Generate date offset for last year
    neg_1_date_field = 'year_neg_1_' + date_field
    new_signals[neg_1_date_field] = new_signals[date_field].apply(
        lambda x: x - timedelta(weeks=lastweeknumberoflastyear(x))
    )
    lag_values, feature_list = lag_features_x_years_ago(
        new_signals, metric, date_field, feature_list, dimension_list, lagged_agg_columns, 1
    )
    new_signals = new_signals.merge(lag_values, on=[neg_1_date_field] + dimension_list, how='left')

    
    # Repeat for 2 years ago
    neg_2_date_field = 'year_neg_2_' + date_field
    new_signals[neg_2_date_field] = new_signals[neg_1_date_field].apply(
        lambda x: x - timedelta(weeks=lastweeknumberoflastyear(x))
    )
    lag_values, feature_list = lag_features_x_years_ago(
        new_signals, metric, date_field, feature_list, dimension_list, lagged_agg_columns, 2
    )
    new_signals = new_signals.merge(lag_values, on=[neg_2_date_field] + dimension_list, how='left')

    # Year-over-year metric ratio
    new_signals['yoy_ratio_value'] = np.where(
        new_signals['year_neg_2_' + metric] > 0,
        (new_signals['year_neg_1_' + metric] - new_signals['year_neg_2_' + metric]) / new_signals['year_neg_2_' + metric],
        np.nan
    )
    feature_list.append('yoy_ratio_value')

    # Optional YOY ratio for week -1 mean
    if 'mean' in agg_list:
        new_signals['yoy_ratio_week_neg_1_mean'] = (
            new_signals['year_neg_1_week_neg_1_mean'] - new_signals['year_neg_2_week_neg_1_mean']
        ) / new_signals['year_neg_2_week_neg_1_mean']
        feature_list.append('yoy_ratio_week_neg_1_mean')

    return new_signals, feature_list


# Add seasonality-based features (month, week, day) and holidays
def seasonality_features(new_signals, metric, date_field, feature_list):
    from datetime import datetime
    import numpy as np
    import math
    import pandas as pd 
    
    # Extract date/time components
    new_signals['year'] = new_signals[date_field].dt.year
    new_signals['month'] = new_signals[date_field].dt.month
    new_signals['quarter'] = new_signals[date_field].dt.quarter
    new_signals['week_of_year'] = new_signals[date_field].dt.isocalendar().week
    new_signals['day_of_year'] = new_signals[date_field].apply(lambda x: (x - datetime(x.year, 1, 1)).days + 1)
    new_signals['day_of_month'] = new_signals[date_field].dt.day
    new_signals['day_of_week'] = new_signals[date_field].dt.dayofweek + 1
    new_signals['weekend'] = new_signals['day_of_week'].isin([6, 7])

    # Add these to feature list
    for i in ['year', 'month', 'week_of_year', 'day_of_year', 'day_of_month', 'day_of_week', 'weekend', 'quarter']:
        feature_list.append(i)

    # Generate cyclical features using sine/cosine transformations
    def generate_cyclical_features(input_df, column_name_input, total):
        sin_column_name = 'sin_' + column_name_input
        cos_column_name = 'cos_' + column_name_input
        input_df[sin_column_name] = np.sin((2 * math.pi) * (input_df[column_name_input] / total))
        input_df[cos_column_name] = np.cos((2 * math.pi) * (input_df[column_name_input] / total))
        feature_list.extend([sin_column_name, cos_column_name])
        return input_df

    for col, total in [('month', 12), ('week_of_year', 52), ('day_of_year', 365), ('day_of_week', 7), ('quarter', 4)]:
        new_signals = generate_cyclical_features(new_signals, col, total)

    # Add categorical representations for season/month/weekday/quarter
    new_signals['season_categorical'] = new_signals.month.map({
        1: 'WINTER', 2: 'WINTER', 3: 'SPRING', 4: 'SPRING', 5: 'SPRING',
        6: 'SUMMER', 7: 'SUMMER', 8: 'SUMMER', 9: 'FALL', 10: 'FALL', 11: 'FALL', 12: 'WINTER'
    })
    new_signals['month_categorical'] = new_signals.month.map(dict(zip(range(1, 13), [
        'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER',
        'OCTOBER', 'NOVEMBER', 'DECEMBER'
    ])))
    new_signals['day_of_week_categorical'] = new_signals.day_of_week.map(dict(zip(range(1, 8), [
        'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY'
    ])))
    new_signals['quarter_categorical'] = new_signals.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

    # One-hot encode categorical columns and add them as features
    for column_name in ['season_categorical', 'month_categorical', 'day_of_week_categorical', 'quarter_categorical']:
        one_hot = pd.get_dummies(new_signals[column_name])
        new_signals = new_signals.join(one_hot)
        feature_list.extend(one_hot.columns)

    # Add public holiday features using the `holidays` library
    import holidays
    year_array = new_signals.year.unique()
    holiday_calendar = {}
    for country_name in ['US', 'UK', 'IE', 'FR', 'IN', 'TR', 'EG', 'CN']:
        holiday_calendar.update(holidays.country_holidays(country_name, years=year_array))

    def check_holidays(dt):
        return holiday_calendar.get(dt.date(), None)

    new_signals['holiday_name'] = new_signals[date_field].apply(check_holidays)
    new_signals['is_public_holiday'] = new_signals['holiday_name'].notnull()
    feature_list.append('is_public_holiday')

    # Add binary features for top holiday names
    TOP_HOLIDAYS = ["New Year's Day", "Christmas Day", "Independence Day", "Labor Day", "Thanksgiving",
                    "Good Friday", "Easter Monday", "Saint Stephen's Day", "All Saints' Day",
                    "Chinese New Year (Spring Festival)", "Diwali", "Eid al-Fitr", "Eid al-Adha", "Republic Day"]
    
    for holiday in TOP_HOLIDAYS:
        reduced_string = holiday.lower().replace(' ', '_').replace("'", '')
        col_name = "is_" + reduced_string
        new_signals[col_name] = new_signals['holiday_name'].apply(
            lambda x: x is not None and holiday.lower() in x.lower()
        )
        feature_list.append(col_name)

    new_signals = new_signals.drop(columns=['holiday_name'])

    # Manually mark key dates
    new_signals['NEW_YEARS_EVE'] = (new_signals['day_of_month'] == 31) & (new_signals['DECEMBER'] == True)
    new_signals['CHRISTMAS_EVE'] = (new_signals['day_of_month'] == 24) & (new_signals['DECEMBER'] == True)
    new_signals['HALLOWEEN'] = (new_signals['day_of_month'] == 31) & (new_signals['OCTOBER'] == True)
    new_signals['VALENTINES'] = (new_signals['day_of_month'] == 14) & (new_signals['FEBRUARY'] == True)

    for i in ['NEW_YEARS_EVE', 'CHRISTMAS_EVE', 'HALLOWEEN', 'VALENTINES']:
        feature_list.append(i)

    return new_signals, feature_list


# One-hot encodes specified categorical dimensions
def dimensional_features(new_signals, metric, date_field, feature_list, cols_to_features):
    for column_name in cols_to_features:
        one_hot = pd.get_dummies(new_signals[column_name])
        new_signals = new_signals.join(one_hot)
        feature_list.extend(one_hot.columns)
    return new_signals, feature_list


# Master function to apply all feature engineering steps
def add_features(input_df, metric, date_field, dimension_list=[], cols_to_features=[]):
    
    feature_list = []
    new_signals = input_df.copy()

    # Ensure data is sorted by datetime
    new_signals = new_signals.sort_values(by=date_field, ascending=True)

    # List of aggregations for lag features
    agg_list = ['mean', 'median', 'min', 'max', 'std']

    # Generate rolling-window lag features from this year
    new_signals, lagged_agg_columns = lag_features_this_year(new_signals, metric, date_field, dimension_list, agg_list)

    # Generate lag features from prior years + YOY signals
    new_signals, feature_list = lag_features(new_signals, metric, date_field, feature_list, dimension_list, lagged_agg_columns, agg_list)

    # Add date-based and holiday-based seasonality features
    new_signals, feature_list = seasonality_features(new_signals, metric, date_field, feature_list)

    # Optionally add one-hot encoded dimensional features (e.g., 'REGION', 'MARKET')
    if len(cols_to_features) > 0:
        new_signals, feature_list = dimensional_features(new_signals, metric, date_field, feature_list, cols_to_features)

    return new_signals, feature_list

