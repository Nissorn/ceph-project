const API_BASE_URL = 'http://localhost:8000/api/v1';

export class ApiClient {
  static async getHealth() {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.json();
  }

  static async analyzeImage(data: any) {
    const res = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return res.json();
  }
}
