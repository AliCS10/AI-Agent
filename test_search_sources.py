import unittest
from unittest.mock import patch, MagicMock
from search_sources import query_variants, kaggle_search, hf_search, unique, rank_results

class TestSources(unittest.TestCase):
    def test_arabic_variants(self):
        result = query_variants('أبغى بيانات مبيعات في السعودية')
        self.assertIn('sales', result[0]); self.assertIn('saudi', result[0])
    @patch('search_sources.requests.get')
    def test_kaggle(self, getter):
        response = MagicMock()
        response.json.return_value = [{'ref':'owner/slug','title':'Coffee Sales','subtitle':'Monthly sales','downloadCount':20,'fileType':'csv','lastUpdated':'2026-01-01'}]
        getter.return_value = response
        out = kaggle_search('coffee')
        self.assertEqual(out[0]['url'], 'https://www.kaggle.com/datasets/owner/slug')
        self.assertEqual(out[0]['source'],'Kaggle')
    @patch('search_sources.requests.get')
    def test_hf(self,getter):
        response=MagicMock(); response.json.return_value=[{'id':'org/arabic-data','downloads':50,'tags':['license:mit'],'lastModified':'2025-03-04T00:00:00Z'}]
        getter.return_value=response
        out=hf_search('arabic')
        self.assertEqual(out[0]['license'],'mit')
        self.assertEqual(out[0]['updated'],'2025-03-04')
    def test_dedupe_rank(self):
        r={'source':'Kaggle','title':'coffee sales','description':'','id':'foo','url':'https://www.kaggle.com/datasets/a/b','updated':'2025-01-01','downloads':10}
        self.assertEqual(len(unique([r,r])),1)
        self.assertEqual(rank_results([r],'coffee','الأكثر صلة')[0]['match_score'],100)

if __name__ == '__main__': unittest.main()
