from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('btc.csv')

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    synthesizer = CTGANSynthesizer(metadata, cuda=True, verbose=True)
    synthesizer.fit(df)

    synthetic_data = synthesizer.sample(num_rows=50000)

    synthetic_data.to_csv('result.csv', index = False, encoding='utf-8')
