def test_predict_page(self):
    sample_text = "Breaking news: Stock markets soar amid economic optimism."
    response = self.client.post('/predict', data=dict(text=sample_text))
    
    print("Response data:", response.data)  # Debugging step
    
    self.assertEqual(response.status_code, 200)
    expected_classes = [b'Business', b'Sports', b'Technology', b'Politics', b'Entertainment']
    
    self.assertTrue(
        any(label in response.data for label in expected_classes),
        f"Response should contain one of the expected classes, but got: {response.data}"
    )

