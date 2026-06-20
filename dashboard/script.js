// script.js – clean async implementation for flux chart and forecast

document.addEventListener('DOMContentLoaded', async () => {
  // Canvas for Chart.js
  const ctx = document.getElementById('fluxChart').getContext('2d');

  // ---------- Theme toggle ----------
  const themeToggle = document.getElementById('themeToggle');
  const applyTheme = (dark) => {
    if (dark) {
      document.body.classList.add('dark');
      themeToggle.checked = true;
    } else {
      document.body.classList.remove('dark');
      themeToggle.checked = false;
    }
    localStorage.setItem('darkMode', dark ? '1' : '0');
  };
  const saved = localStorage.getItem('darkMode');
  applyTheme(saved === '1');
  themeToggle.addEventListener('change', () => applyTheme(themeToggle.checked));

  // Show spinner while loading
  const loadingSpinner = document.getElementById('loadingSpinner');
  loadingSpinner.style.display = 'block';

  // ---------- Load observed flux ----------
  try {
    const csvResp = await fetch('/data/merged.csv'); // absolute path from Flask root
    if (!csvResp.ok) throw new Error('Network response was not ok');
    const csvText = await csvResp.text();
    const rows = csvText.trim().split('\n').slice(1); // skip header
    const labels = [];
    const counts = [];
    rows.forEach(row => {
      const [date, countStr, time] = row.split(',');
      const timestamp = time ? time : date;
      labels.push(timestamp);
      const cnt = parseFloat(countStr);
      counts.push(isNaN(cnt) ? 0 : cnt);
    });
    new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Observed Flux (Counts)',
          data: counts,
          borderColor: '#4fd1c5',
          backgroundColor: 'rgba(79,209,197,0.2)',
          pointRadius: 0,
          tension: 0.2,
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: { display: true, title: { display: true, text: 'Time' } },
          y: { display: true, title: { display: true, text: 'Counts' } },
        },
      },
    });
  } catch (err) {
    document.getElementById('forecastResult').textContent = `Error loading data: ${err.message}`;
  }

  // ---------- Load forecast ----------
  try {
    const resp = await fetch('/forecast');
    if (!resp.ok) throw new Error('Network response was not ok');
    const data = await resp.json();
    const container = document.getElementById('forecastResult');
    const cls = data.class || 'N/A';
    const flux = data.peak_flux_W_m2 !== undefined ? data.peak_flux_W_m2 : 'N/A';
    container.innerHTML = `<strong>Forecast:</strong> Class ${cls}, Peak Flux ${flux} W/m²`;
  } catch (err) {
    document.getElementById('forecastResult').textContent = `Error loading forecast: ${err.message}`;
  }

  loadingSpinner.style.display = 'none';
});

  // Load saved theme
  const saved = localStorage.getItem('darkMode');
  applyTheme(saved === '1');
  themeToggle.addEventListener('change', () => {
    applyTheme(themeToggle.checked);
  });

  fetch('../data/merged.csv')
    .then(response => {
      if (!response.ok) throw new Error('Network response was not ok');
      return response.text();
    })
    .then(csvText => {
      const rows = csvText.trim().split('\n').slice(1); // skip header line
      const labels = [];
      const counts = [];
      rows.forEach(row => {
        const [date, countsStr, time] = row.split(',');
        // Use the provided time field if present, otherwise the DATE column
        const timestamp = time ? time : date;
        labels.push(timestamp);
        const count = parseFloat(countsStr);
        counts.push(isNaN(count) ? 0 : count);
      });
      new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Observed Flux (Counts)',
            data: counts,
            borderColor: '#4fd1c5',
            backgroundColor: 'rgba(79,209,197,0.2)',
            pointRadius: 0,
            tension: 0.2,
          }]
        },
        options: {
          responsive: true,
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Time'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Counts'
              }
            }
          }
        }
      });

  // Show loading spinner
  document.getElementById('loadingSpinner').style.display = 'block';
  // Fetch real forecast from backend
  fetch('/forecast')
    .then(r => { if (!r.ok) throw new Error('Network response was not ok'); return r.json(); })
    .then(data => {
      const container = document.getElementById('forecastResult');
      const cls = data.class || 'N/A';
      const flux = data.peak_flux_W_m2 !== undefined ? data.peak_flux_W_m2 : 'N/A';
      container.innerHTML = `<strong>Forecast:</strong> Class ${cls}, Peak Flux ${flux} W/m²`;
    })
    .catch(err => {
      const container = document.getElementById('forecastResult');
      container.textContent = `Error loading forecast: ${err.message}`;
    })
    .finally(() => {
      document.getElementById('loadingSpinner').style.display = 'none';
    });
    })
    .catch(err => {
      const container = document.getElementById('forecastResult');
      container.textContent = `Error loading data: ${err.message}`;
    });
});
