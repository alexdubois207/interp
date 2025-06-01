import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline, Akima1DInterpolator, PchipInterpolator, Rbf, griddata, bisplrep, bisplev, CloughTocher2DInterpolator
from sklearn.metrics import mean_squared_error


def interpolate_1d(train, method_name):
    results = []
    for exp in train['exp_y'].unique():
        subset = train[train['exp_y'] == exp]

        # First interpolate across strikes (within each fixed maturity)
        intermediate = []
        for mat in subset['mat_y'].unique():
            sub = subset[subset['mat_y'] == mat]
            known = sub.dropna(subset=['vol_ts'])
            if len(known) < 2:
                continue
            try:
                x = known['strike'].values
                y = known['vol_ts'].values
                if method_name == 'linear':
                    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
                elif method_name == 'polynomial':
                    f = np.poly1d(np.polyfit(x, y, deg=min(3, len(x)-1)))
                elif method_name == 'cubic':
                    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                elif method_name == 'cubic_spline':
                    f = CubicSpline(x, y, extrapolate=True)
                elif method_name == 'akima':
                    f = Akima1DInterpolator(x, y)
                elif method_name == 'hermite':
                    f = PchipInterpolator(x, y)
                elif method_name == 'rbf':
                    f = Rbf(x, y, y, function='linear')
                else:
                    continue
                sub_interp = sub.copy()
                sub_interp['vol_tmp'] = f(sub['strike'].values)
                intermediate.append(sub_interp)
            except Exception as e:
                continue

        if not intermediate:
            continue

        interpolated_strike = pd.concat(intermediate, ignore_index=True)

        # Now interpolate across maturities (for each fixed strike)
        for strike_val in interpolated_strike['strike'].unique():
            sub = interpolated_strike[interpolated_strike['strike'] == strike_val]
            known = sub.dropna(subset=['vol_tmp'])
            if len(known) < 2:
                continue
            try:
                x = known['mat_y'].values
                y = known['vol_tmp'].values
                if method_name == 'linear':
                    f = interp1d(x, y, kind='linear', fill_value="extrapolate")
                elif method_name == 'polynomial':
                    f = np.poly1d(np.polyfit(x, y, deg=min(3, len(x)-1)))
                elif method_name == 'cubic':
                    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                elif method_name == 'cubic_spline':
                    f = CubicSpline(x, y, extrapolate=True)
                elif method_name == 'akima':
                    f = Akima1DInterpolator(x, y)
                elif method_name == 'hermite':
                    f = PchipInterpolator(x, y)
                elif method_name == 'rbf':
                    f = Rbf(x, y, y, function='linear')
                else:
                    continue

                sub_interp = sub.copy()
                sub_interp['vol_' + method_name + '_1d'] = f(sub['mat_y'].values)
                results.append(sub_interp[['mat_y', 'exp_y', 'strike', 'vol_' + method_name + '_1d']])
            except Exception as e:
                continue

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def interpolate_2d(train, method_name):
    results = []
    for exp in train['exp_y'].unique():
        subset = train[train['exp_y'] == exp]
        known = subset.dropna(subset=['vol_ts'])
        if len(known) < 4:
            continue
        try:
            x = known['mat_y'].values
            y = known['strike'].values
            z = known['vol_ts'].values
            grid_x = subset['mat_y'].values
            grid_y = subset['strike'].values

            if method_name == 'bilinear':
                method = 'linear'
                z_interp = griddata((x, y), z, (grid_x, grid_y), method=method)
            elif method_name == 'bicubic':
                method = 'cubic'
                z_interp = griddata((x, y), z, (grid_x, grid_y), method=method)
            elif method_name == 'rbf':
                rbf = Rbf(x, y, z, function='linear')
                z_interp = rbf(grid_x, grid_y)
            elif method_name == 'bivariate_spline':
                tck = bisplrep(x, y, z)
                z_interp = bisplev(grid_x, grid_y, tck)
            elif method_name == 'clough':
                ct_interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
                z_interp = ct_interp(grid_x, grid_y)
            else:
                continue

            result = subset.copy()
            result['vol_' + method_name + '_2d'] = z_interp
            results.append(result[['mat_y', 'exp_y', 'strike', 'vol_' + method_name + '_2d']])
        except Exception as e:
            continue

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def evaluate_methods(train, interpolated_dfs):
    evaluations = {}
    for method_df in interpolated_dfs:
        common = train.merge(method_df, on=['mat_y', 'exp_y', 'strike'], how='left')
        interp_cols = [col for col in common.columns if col.startswith('vol_') and col != 'vol_test']
        for col in interp_cols:
            valid = common[~common['vol_test'].isna() & ~common[col].isna()]
            mse = mean_squared_error(valid['vol_test'], valid[col])
            evaluations[col] = mse
    return evaluations

methods_1d = ['linear', 'polynomial', 'cubic', 'cubic_spline', 'akima', 'hermite', 'rbf']
methods_2d = ['bilinear', 'bicubic', 'rbf', 'bivariate_spline', 'clough']

interpolated_dfs = []

for method in methods_1d:
    df = interpolate_1d(train, method)
    if not df.empty:
        train = train.merge(df, on=['mat_y', 'exp_y', 'strike'], how='left')
        interpolated_dfs.append(df)

for method in methods_2d:
    df = interpolate_2d(train, method)
    if not df.empty:
        train = train.merge(df, on=['mat_y', 'exp_y', 'strike'], how='left')
        interpolated_dfs.append(df)

# Evaluate
evaluation_results = evaluate_methods(train, interpolated_dfs)
evaluation_results
