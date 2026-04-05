document.addEventListener('DOMContentLoaded', () => {

    // --- Navigation System ---
    const navLinks = document.querySelectorAll('.nav-links li');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));

            e.target.classList.add('active');
            const targetId = e.target.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');

            if (targetId === 'insights' && !chartsLoaded) {
                loadInsights();
            } else if (targetId === 'filtered' && !filtersLoaded) {
                applyFilters(); // Initial load
            } else if (targetId === 'inventory') {
                loadInventory();
            }
        });
    });

    // --- Sales Prediction Logic ---
    const predictForm = document.getElementById('predict-form');
    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(predictForm);
        const data = Object.fromEntries(formData.entries());

        Object.keys(data).forEach(k => {
            if (k !== 'day_of_week') {
                data[k] = Number(data[k]);
            }
        });

        const btn = predictForm.querySelector('button');
        btn.innerText = "Predicting...";

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await res.json();

            document.getElementById('prediction-result').classList.remove('hidden');
            const valEl = document.getElementById('prediction-value');
            valEl.innerText = `$${result.prediction.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        } catch (error) {
            console.error(error);
            alert("Error fetching prediction");
        } finally {
            btn.innerText = "Predict Expected Sales 🚀";
        }
    });

    // --- Insights Loading ---
    let chartsLoaded = false;
    async function loadInsights() {
        try {
            const res = await fetch('/api/insights/overall');
            const charts = await res.json();

            const layoutConfig = {
                margin: { t: 40, l: 40, r: 20, b: 40 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };

            Plotly.newPlot('chart-salesTrend', charts.salesTrend.data, { ...charts.salesTrend.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-store', charts.store.data, { ...charts.store.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-promo', charts.promo.data, { ...charts.promo.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-assort', charts.assort.data, { ...charts.assort.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-day', charts.dayOfWeek.data, { ...charts.dayOfWeek.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-stateHol', charts.stateHol.data, { ...charts.stateHol.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-schoolHol', charts.schoolHol.data, { ...charts.schoolHol.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-combo', charts.combo.data, { ...charts.combo.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-monthlySh', charts.monthlySh.data, { ...charts.monthlySh.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-monthlyPromo', charts.monthlyPromo.data, { ...charts.monthlyPromo.layout, ...layoutConfig }, { responsive: true });
            Plotly.newPlot('chart-corr', charts.corr.data, { ...charts.corr.layout, ...layoutConfig }, { responsive: true });

            chartsLoaded = true;
        } catch (error) {
            console.error("Failed to load insights:", error);
        }
    }

    // --- Filtered Insights System ---
    let filtersLoaded = false;
    const filterForm = document.getElementById('filter-form');
    filterForm.addEventListener('submit', (e) => {
        e.preventDefault();
        applyFilters();
    });

    async function applyFilters() {
        const payload = {
            store_type: document.getElementById('filt_store').value.split(',').map(s => s.trim()),
            assortment: document.getElementById('filt_assort').value.split(',').map(s => s.trim()),
            school_holiday: document.getElementById('filt_sh').value,
            months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], // Kept simple for UI demo
            days: document.getElementById('filt_days').value.split(',').map(s => s.trim()),
            exclude_zero_sales: document.getElementById('filt_exclude').checked
        };

        const btn = filterForm.querySelector('button');
        btn.innerText = "Processing...";

        try {
            const res = await fetch('/api/insights/filtered', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            document.getElementById('filter-metrics').innerHTML = `
                <div class="metric-card"><div class="metric-label">Qualified Stores</div><div class="metric-value txt-primary">${data.metrics.stores}</div></div>
                <div class="metric-card"><div class="metric-label">Avg Sales (Filtered)</div><div class="metric-value txt-success">$${data.metrics.avg_sales.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div></div>
                <div class="metric-card"><div class="metric-label">Dataset Records</div><div class="metric-value">${data.metrics.records.toLocaleString()}</div></div>
            `;

            if (data.charts) {
                const layoutConfig = { margin: { t: 40, l: 40, r: 20, b: 40 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };
                Plotly.newPlot('chart-filt-dist', data.charts.dist.data, { ...data.charts.dist.layout, ...layoutConfig }, { responsive: true });
                Plotly.newPlot('chart-filt-promo', data.charts.promo2.data, { ...data.charts.promo2.layout, ...layoutConfig }, { responsive: true });
            } else {
                document.getElementById('chart-filt-dist').innerHTML = "<p>No data matches the filter.</p>";
                document.getElementById('chart-filt-promo').innerHTML = "";
            }

            filtersLoaded = true;
        } catch (error) {
            console.error("Filter error:", error);
        } finally {
            btn.innerText = "Apply Filters";
        }
    }


    // --- Inventory System Logic ---
    const invTbody = document.getElementById('inventory-tbody');
    const invForm = document.getElementById('inventory-form');
    const metricsDiv = document.getElementById('inventory-metrics');

    async function loadInventory() {
        try {
            const res = await fetch('/api/inventory');
            const data = await res.json();

            renderInventoryMetrics(data);

            invTbody.innerHTML = '';
            data.forEach(item => {
                const tr = document.createElement('tr');
                const isLow = item.Stock <= item.Reorder_Level;
                const rowStyle = isLow ? 'background-color: rgba(239,68,68,0.1)' : '';
                tr.innerHTML = `
                    <td style="${rowStyle}">${item.Item_ID}</td>
                    <td style="${rowStyle}"><strong>${item.Item_Name}</strong></td>
                    <td style="${rowStyle}">${item.Category}</td>
                    <td style="${rowStyle}; color: ${isLow ? '#ef4444' : 'inherit'}; font-weight: ${isLow ? 'bold' : 'normal'}">${item.Stock}</td>
                    <td style="${rowStyle}">${item.Reorder_Level}</td>
                    <td style="${rowStyle}">$${item.Unit_Price.toFixed(2)}</td>
                    <td style="${rowStyle}">
                        <button class="btn-danger" onclick="deleteItem(${item.Item_ID})">Del</button>
                    </td>
                `;
                tr.style.cursor = "pointer";
                tr.title = "Click to edit";
                tr.onclick = (e) => {
                    if (e.target.tagName !== 'BUTTON') {
                        document.getElementById('inv_id').value = item.Item_ID;
                        document.getElementById('inv_name').value = item.Item_Name;
                        document.getElementById('inv_cat').value = item.Category;
                        document.getElementById('inv_stock').value = item.Stock;
                        document.getElementById('inv_reorder').value = item.Reorder_Level;
                        document.getElementById('inv_price').value = item.Unit_Price;
                    }
                };
                invTbody.appendChild(tr);
            });
        } catch (error) {
            console.error("Inventory load error:", error);
        }
    }

    function renderInventoryMetrics(data) {
        const totalItems = data.length;
        const lowAuth = data.filter(d => d.Stock <= d.Reorder_Level).length;
        const totalValue = data.reduce((sum, d) => sum + (d.Stock * d.Unit_Price), 0);

        metricsDiv.innerHTML = `
            <div class="metric-card">
                <div class="metric-label">Total Unique Items</div>
                <div class="metric-value txt-primary">${totalItems}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Low Stock Alerts</div>
                <div class="metric-value txt-danger">${lowAuth}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Asset Value</div>
                <div class="metric-value txt-success">$${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
            </div>
        `;
    }

    invForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        let idVal = document.getElementById('inv_id').value;
        if (!idVal) idVal = Math.floor(Math.random() * 10000) + 2000;

        const payload = {
            Item_ID: Number(idVal),
            Item_Name: document.getElementById('inv_name').value,
            Category: document.getElementById('inv_cat').value,
            Stock: Number(document.getElementById('inv_stock').value),
            Reorder_Level: Number(document.getElementById('inv_reorder').value),
            Unit_Price: Number(document.getElementById('inv_price').value)
        };

        try {
            await fetch('/api/inventory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            invForm.reset();
            loadInventory();
        } catch (err) {
            alert("Error saving item");
        }
    });

    window.deleteItem = async function (id) {
        if (confirm(`Are you sure you want to delete item ${id}?`)) {
            await fetch(`/api/inventory/${id}`, { method: 'DELETE' });
            loadInventory();
        }
    };
});
