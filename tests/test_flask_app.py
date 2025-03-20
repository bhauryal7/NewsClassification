import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>News classification</title>', response.data)

    def test_predict_page(self):
        sample_text = "Breaking news: Stock markets soar amid economic optimism."
        response = self.client.post('/predict', data=dict(text=sample_text))
        
        self.assertEqual(response.status_code, 200)
        expected_classes = ['ENTERTAINMENT','HEALTH','TECHNOLOGY','WORLD','BUSINESS','SPORTS','NATION','SCIENCE'] 
        
        self.assertTrue(
            any(label in response.data for label in expected_classes),
            "Response should contain one of the expected classes."
        )

if __name__ == '__main__':
    unittest.main()




['ENTERTAINMENT',
 'HEALTH',
 'TECHNOLOGY',
 'WORLD',
 'BUSINESS',
 'SPORTS',
 'NATION',
 'SCIENCE']    