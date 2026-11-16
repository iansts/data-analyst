"""Deterministic tools. The LLM may propose changes, never approve them."""
from __future__ import annotations

import copy
import csv
import io
import json
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


def load_table_auto(data: bytes, name: str):
    """Detect common CSV settings; Excel opens its first worksheet."""
    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Максимальний розмір файлу — 10 МБ.')
    if name.lower().endswith('.xlsx'):
        return load_table(data, name)
    try:
        text = data.decode('utf-8-sig')
        encoding = 'utf-8-sig'
    except UnicodeDecodeError:
        text = data.decode('cp1251')
        encoding = 'cp1251'
    try:
        sep = csv.Sniffer().sniff(text[:16384], delimiters=',;\t|').delimiter
    except csv.Error:
        sep = ','
    return load_table(data, name, sep=sep, encoding=encoding)


def records(df, limit=30):
    return json.loads(df.head(limit).to_json(orient='records', date_format='iso', force_ascii=False))


def profile(df):
    return {'rows': len(df), 'columns': len(df.columns),
            'duplicate_rows': int(df.duplicated().sum()),
            'missing_cells': int(df.isna().sum().sum()),
            'fields': [{'name': c, 'type': str(df[c].dtype),
                        'missing': int(df[c].isna().sum()),
                        'unique': int(df[c].nunique()),
                        'examples': [str(x)[:160] for x in df[c].dropna().unique()[:5]]}
                       for c in df.columns]}


def load_table(data: bytes, name: str, sheet=0, sep=',', encoding='utf-8-sig'):
    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Максимальний розмір файлу — 10 МБ.')
    if name.lower().endswith('.xlsx'):
        # Read-only openpyxl avoids expanding all sheets; values only, no macros.
        from openpyxl import load_workbook
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            ws = book[book.sheetnames[sheet] if isinstance(sheet, int) else sheet]
            rows = ws.iter_rows(values_only=True)
            header = next(rows, None)
            if not header or len(header) > 60:
                raise ValueError('Порожня таблиця або понад 60 колонок.')
            names = [str(c) if c is not None else f'column_{i+1}' for i, c in enumerate(header)]
            if len(set(names)) != len(names):
                raise ValueError('Назви колонок мають бути унікальними.')
            values = []
            for row in rows:
                values.append(row)
                if len(values) > 20000:
                    raise ValueError('Ліміт — 20 000 рядків.')
            df = pd.DataFrame(values, columns=names)
        finally:
            book.close()
    else:
        df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding, nrows=20001)
    df.columns = df.columns.astype(str)
    if df.empty or len(df) > 20000 or len(df.columns) > 60:
        raise ValueError('Потрібно 1–20 000 рядків і не більше 60 колонок.')
    return df.reset_index(drop=True)


def transform(source, operations):
    if not isinstance(operations, list) or not 1 <= len(operations) <= 15:
        raise ValueError('План має містити 1–15 операцій.')
    df = source.copy(deep=True)
    for op in operations:
        kind = op['kind']
        col = op.get('column')
        if kind not in ('drop_duplicates', 'drop_rows_where') and col not in df.columns:
            raise ValueError(f'Невідома колонка: {col}')
        if kind == 'trim':
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        elif kind == 'replace':
            mapping = op.get('mapping', {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('Потрібен непорожній mapping.')
            df[col] = df[col].replace(mapping)
        elif kind == 'fill_missing':
            if 'value' not in op or op['value'] is None:
                raise ValueError('Вкажіть значення заповнення.')
            df[col] = df[col].astype(object).where(df[col].notna(), op['value'])
        elif kind == 'to_number':
            s = df[col].astype('string').str.strip().str.replace('\u00a0', '', regex=False).str.replace(' ', '', regex=False)
            decimal = op.get('decimal', '.')
            if decimal not in ('.', ','):
                raise ValueError('decimal має бути . або ,')
            if decimal == ',':
                s = s.str.replace(',', '.', regex=False)
            converted = pd.to_numeric(s, errors='coerce')
            bad = df[col].notna() & converted.isna()
            if bad.any():
                raise ValueError(f'{col}: {int(bad.sum())} значень не є числами; спочатку узгодьте їх заміну.')
            df[col] = converted
        elif kind == 'to_date':
            fmt = op.get('format')
            if not fmt or fmt in ('mixed', 'ISO8601'):
                raise ValueError('Потрібен явний формат, наприклад %d.%m.%Y.')
            converted = pd.to_datetime(df[col], format=fmt, errors='coerce')
            if (df[col].notna() & converted.isna()).any():
                raise ValueError(f'{col}: є дати, що не відповідають формату {fmt}.')
            df[col] = converted.dt.strftime('%Y-%m-%d').where(converted.notna(), None)
        elif kind == 'drop_rows_where':
            value = op.get('value')
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError('Вкажіть текстове, числове або логічне значення для видалення рядків.')
            if 'column' in op and (not isinstance(col, str) or col not in df.columns):
                raise ValueError(f'Невідома колонка: {col}')
            columns = df[[col]] if 'column' in op else df
            matches = columns.eq(value).fillna(False)
            if not isinstance(value, str):
                # Python treats True / False as equal to 1 / 0. Keep them distinct.
                matches &= columns.map(pd.api.types.is_bool).eq(isinstance(value, bool))
            df = df.loc[~matches.any(axis=1)]
        elif kind == 'drop_duplicates':
            subset = op.get('subset') or None
            if subset and (not isinstance(subset, list) or any(c not in df.columns for c in subset)):
                raise ValueError('Невірні колонки для пошуку дублів.')
            df = df.drop_duplicates(subset=subset, keep='first')
        else:
            raise ValueError(f'Непідтримувана операція: {kind}')
    return df


class Clinic:
    def __init__(self, df):
        self.original = df.copy(deep=True)
        self.df = df.copy(deep=True)
        self.revision = 0
        self.pending = None
        self.snapshots = []
        self.audit = []

    def log(self, action, details):
        self.audit.append({'time': datetime.now(timezone.utc).isoformat(),
                           'action': action, 'revision': self.revision, 'details': details})

    def inspect_rows(self, column='', query='', limit=20):
        limit = max(1, min(int(limit), 30))
        df = self.df
        if column:
            if column not in df:
                raise ValueError('Невідома колонка.')
            df = df[df[column].astype(str).str.contains(str(query), case=False, regex=False)]
        return {'matched': len(df), 'sample': records(df, limit)}

    def check_column(self, column):
        s = self.df[column]
        numbers = pd.to_numeric(s, errors='coerce')
        return {'column': column, 'missing': int(s.isna().sum()),
                'whitespace': int(s.map(lambda x: isinstance(x, str) and x != x.strip()).sum()),
                'non_numeric': int((s.notna() & numbers.isna()).sum()),
                'negative_numbers': int((numbers < 0).sum()),
                'most_frequent': {str(k)[:160]: int(v) for k,v in s.value_counts().head(15).items()}}

    def aggregate(self, group_by, value, method='sum'):
        if method not in ('sum', 'mean', 'count', 'min', 'max'):
            raise ValueError('Невідомий метод.')
        if method != 'count' and not pd.api.types.is_numeric_dtype(self.df[value]):
            raise ValueError('Спочатку перетворіть колонку на числа через погоджений план.')
        result = self.df.groupby(group_by, dropna=False)[value].agg(method).reset_index()
        return {'groups': len(result), 'sample': records(result)}

    def propose_plan(self, operations, explanation):
        candidate = transform(self.df, operations)
        before = self.df.loc[candidate.index]
        equal = before.eq(candidate) | (before.isna() & candidate.isna())
        changed = ~equal.all(axis=1)
        indexes = candidate.index[changed].tolist()[:30]
        removed = self.df.loc[~self.df.index.isin(candidate.index)]
        preview = {'id': uuid4().hex[:12], 'revision': self.revision,
                   'explanation': explanation, 'operations': copy.deepcopy(operations),
                   'before': profile(self.df), 'after': profile(candidate),
                   'changed_rows': int(changed.sum()), 'removed_rows': len(removed),
                   'before_sample': records(self.df.loc[indexes]),
                   'after_sample': records(candidate.loc[indexes]),
                   'removed_sample': records(removed)}
        self.pending = {'preview': preview, 'candidate': candidate}
        return preview

    def approve(self, plan_id):
        if not self.pending or self.pending['preview']['id'] != plan_id:
            raise ValueError('План більше не актуальний.')
        if self.pending['preview']['revision'] != self.revision:
            raise ValueError('Дані змінилися; потрібен новий план.')
        self.snapshots.append(self.df.copy(deep=True))
        self.snapshots = self.snapshots[-5:]
        self.df = self.pending['candidate'].copy(deep=True)
        self.revision += 1
        self.log('approved_plan', self.pending['preview'])
        self.pending = None
        return profile(self.df)

    def reject(self):
        if self.pending:
            self.log('rejected_plan', self.pending['preview'])
        self.pending = None

    def undo(self):
        if not self.snapshots:
            raise ValueError('Немає версій для відкату.')
        self.df = self.snapshots.pop()
        self.pending = None
        self.revision += 1
        self.log('undo', profile(self.df))

    def report(self):
        return {'original': profile(self.original), 'current': profile(self.df),
                'revision': self.revision, 'audit': self.audit}


def export_csv(df):
    # Neutralize spreadsheet formula injection when the CSV is opened in Excel.
    safe = df.copy()
    for c in safe:
        safe[c] = safe[c].map(lambda x: "'" + x if isinstance(x, str) and
                              x.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')


def export_xlsx(df):
    from openpyxl import Workbook
    book = Workbook()
    ws = book.active
    ws.title = 'Cleaned data'
    ws.append(list(df.columns))
    for row in df.itertuples(index=False, name=None):
        ws.append([None if pd.isna(v) else v for v in row])
    for row in ws:
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = 's'
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()
