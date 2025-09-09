// App.js
import React, { useEffect, useState } from 'react';
import { View, Text, ActivityIndicator, FlatList, StyleSheet } from 'react-native';
import * as Location from 'expo-location';
import axios from 'axios';

export default function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [recommendations, setRecs] = useState([]);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setError('Location permission denied');
        setLoading(false);
        return;
      }

      try {
        const { coords } = await Location.getCurrentPositionAsync({});
        const { latitude: lat, longitude: lon } = coords;

        const reverseUrl = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`;
        const geoRes = await fetch(reverseUrl, {
          headers: {
            'User-Agent': 'CropAdvisorApp/1.0 (soumyadipbanerjee6969@gmail.com)'
          }
        });
        if (!geoRes.ok) {
          throw new Error(`Reverse geocode failed: ${geoRes.status}`);
        }
        const geoData = await geoRes.json();
        const district = geoData.address.state_district || geoData.address.county;
        if (!district) {
          throw new Error('District not found via reverse geocoding');
        }

        const apiUrl = 'http://192.168.29.235:8000/recommend';
        const apiRes = await axios.get(apiUrl, { params: { district, lat, lon } });
        if (apiRes.data.error) {
          throw new Error(apiRes.data.error);
        }

        setSummary(apiRes.data.summary);
        setRecs(apiRes.data.recommendations);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return <ActivityIndicator style={styles.centered} size="large" />;
  }
  if (error) {
    return (
      <View style={styles.centered}>
        <Text style={styles.error}>{error}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Top Recommendation</Text>
      <Text style={styles.topCrop}>
        {summary.top_recommended_crop} — {summary.top_recommendation_score}%
      </Text>
      <FlatList
        data={recommendations}
        keyExtractor={(item) => item.crop + item.season}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Text style={styles.crop}>{item.crop} ({item.season})</Text>
            <Text>Score: {item.combined_score}% — {item.recommendation_level}</Text>
            <Text>₹{item.mandi_price}/q, Rev: ₹{item.expected_revenue}/acre</Text>
            <Text>Water: {item.water_requirement}, {item.growth_duration_days} days</Text>
          </View>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: '#f0f0f0' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  error: { color: 'red', fontSize: 16 },
  header: { fontSize: 24, fontWeight: 'bold', marginBottom: 8 },
  topCrop: { fontSize: 20, marginBottom: 16, color: '#006600' },
  card: { backgroundColor: '#fff', padding: 12, marginVertical: 6, borderRadius: 4 },
  crop: { fontSize: 18, fontWeight: '600' }
});
