import json
from pathlib import Path
from jinja2 import Template
from core.evaluator import EvaluationRunSummary
from loguru import logger

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Quality Evaluation Dashboard</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <!-- Tailwind CSS (Direct CDN) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Outfit', 'sans-serif'],
                        mono: ['Space Grotesk', 'monospace'],
                    }
                }
            }
        }
    </script>
    <style>
        body {
            background-color: #0b0f19;
            color: #f8fafc;
            background-image: 
                radial-gradient(at 0% 0%, rgba(20, 30, 55, 0.5) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(30, 15, 60, 0.4) 0px, transparent 50%);
            background-attachment: fixed;
        }
        .glass-card {
            background: rgba(25, 30, 50, 0.55);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.2);
            box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.3);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.3);
            border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.5);
        }
    </style>
</head>
<body class="min-h-screen py-8 px-4 sm:px-6 lg:px-8 font-sans">
    <div class="max-w-7xl mx-auto space-y-8">
        
        <!-- Header -->
        <header class="glass-card rounded-2xl p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="h-4 w-4 rounded-full bg-indigo-500 animate-pulse shadow-[0_0_12px_rgba(99,102,241,0.5)]"></span>
                    <h1 class="text-3xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">AI Quality Evaluation</h1>
                </div>
                <p class="text-sm text-slate-400 mt-1">Prompt Regression & Quality Analytics Framework</p>
            </div>
            
            <div class="flex flex-wrap gap-3 text-sm">
                <div class="px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
                    <span class="text-slate-400">Model:</span>
                    <span class="font-semibold text-indigo-400 ml-1">{{ summary.model }}</span>
                </div>
                <div class="px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
                    <span class="text-slate-400">Run ID:</span>
                    <span class="font-mono text-indigo-400 ml-1">{{ summary.run_id }}</span>
                </div>
                <div class="px-4 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
                    <span class="text-slate-400">Time:</span>
                    <span class="text-slate-200 ml-1">{{ summary.timestamp }}</span>
                </div>
            </div>
        </header>

        <!-- KPI Summary Cards -->
        <section class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <!-- Pass Rate -->
            <div class="glass-card rounded-2xl p-6 flex items-center justify-between">
                <div>
                    <p class="text-xs text-slate-400 uppercase tracking-wider">Overall Pass Rate</p>
                    <h3 class="text-3xl font-bold mt-2 text-emerald-400">{{ (summary.pass_rate * 100) | round(1) }}%</h3>
                    <p class="text-xs text-slate-500 mt-1">{{ summary.passed_count }} passed / {{ summary.total_test_cases }} total</p>
                </div>
                <div class="relative h-14 w-14">
                    <!-- SVG Circle Progress -->
                    <svg class="h-full w-full" viewBox="0 0 36 36">
                        <path class="text-slate-800" stroke-width="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                        <path class="text-emerald-400" stroke-dasharray="{{ (summary.pass_rate * 100) | round }}, 100" stroke-width="3" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    </svg>
                </div>
            </div>

            <!-- Latency -->
            <div class="glass-card rounded-2xl p-6">
                <p class="text-xs text-slate-400 uppercase tracking-wider">Average Latency</p>
                <h3 class="text-3xl font-bold mt-2 text-indigo-400">{{ summary.avg_latency }}s</h3>
                <p class="text-xs text-slate-500 mt-1">Total: {{ summary.total_latency }}s</p>
            </div>

            <!-- Tokens Used -->
            <div class="glass-card rounded-2xl p-6">
                <p class="text-xs text-slate-400 uppercase tracking-wider">Token Count</p>
                <h3 class="text-3xl font-bold mt-2 text-purple-400">{{ "{:,}".format(summary.total_tokens) }}</h3>
                <p class="text-xs text-slate-500 mt-1">Input + Completion</p>
            </div>

            <!-- Cost -->
            <div class="glass-card rounded-2xl p-6">
                <p class="text-xs text-slate-400 uppercase tracking-wider">Estimated Cost</p>
                <h3 class="text-3xl font-bold mt-2 text-pink-400">${{ "{:.6f}".format(summary.total_cost) }}</h3>
                <p class="text-xs text-slate-500 mt-1">Based on token rates</p>
            </div>
        </section>

        <!-- Charts Section -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-semibold text-slate-200 mb-4">Average Quality Scores</h3>
                <div id="metricsBarChart" class="h-72 w-full"></div>
            </div>
            
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-semibold text-slate-200 mb-4">Latency vs Cost Distribution</h3>
                <div id="latencyCostScatter" class="h-72 w-full"></div>
            </div>
        </section>

        <!-- Failure Analysis & Category Aggregations -->
        <section class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="glass-card rounded-2xl p-6 md:col-span-2">
                <h3 class="text-lg font-semibold text-slate-200 mb-4">Category Metrics Summary</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm text-slate-300">
                        <thead>
                            <tr class="border-b border-slate-800 text-xs uppercase text-slate-400 tracking-wider">
                                <th class="pb-3 font-semibold">Category</th>
                                <th class="pb-3 font-semibold">Count</th>
                                <th class="pb-3 font-semibold">Pass Rate</th>
                                <th class="pb-3 font-semibold">Avg Latency</th>
                                <th class="pb-3 font-semibold">Avg Cost</th>
                            </tr>
                        </thead>
                        <tbody id="categoryTableBody" class="divide-y divide-slate-800/40">
                            <!-- Populated dynamically by JS -->
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-semibold text-slate-200 mb-4">Run Status</h3>
                <div id="statusPieChart" class="h-56 w-full"></div>
            </div>
        </section>

        <!-- Filters and Search -->
        <section class="glass-card rounded-2xl p-6 space-y-4">
            <h3 class="text-lg font-semibold text-slate-200">Detailed Test Results</h3>
            
            <div class="flex flex-col md:flex-row gap-4 items-center">
                <!-- Search Box -->
                <div class="relative w-full md:w-80">
                    <input type="text" id="searchInput" placeholder="Search tests..." class="w-full px-4 py-2 pl-10 rounded-xl bg-slate-950/80 border border-slate-850 text-sm text-slate-100 focus:outline-none focus:border-indigo-500">
                    <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
                        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                        </svg>
                    </div>
                </div>

                <!-- Category Filters -->
                <div class="flex flex-wrap gap-2 items-center w-full md:w-auto">
                    <button class="filter-btn active px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-500 text-white" onclick="filterCategory('ALL')">All Categories</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800" onclick="filterCategory('hallucination')">Hallucination</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800" onclick="filterCategory('context_retention')">Context Retention</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800" onclick="filterCategory('relevance')">Relevance</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800" onclick="filterCategory('completeness')">Completeness</button>
                    <button class="filter-btn px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800" onclick="filterCategory('consistency')">Consistency</button>
                </div>

                <!-- Status Filter -->
                <div class="w-full md:w-auto md:ml-auto">
                    <select id="statusFilter" class="w-full md:w-36 px-3 py-2 rounded-xl bg-slate-950/80 border border-slate-850 text-sm text-slate-300 focus:outline-none" onchange="filterStatus()">
                        <option value="ALL">All Statuses</option>
                        <option value="PASSED">Passed</option>
                        <option value="FAILED">Failed</option>
                    </select>
                </div>
            </div>
        </section>

        <!-- Test Cases Accordions -->
        <section id="testCasesContainer" class="space-y-4">
            <!-- Populated dynamically via JavaScript for responsiveness -->
        </section>

    </div>

    <!-- Data Injection -->
    <script id="summary-data" type="application/json">
        {{ summary_json }}
    </script>

    <script>
        const runData = JSON.parse(document.getElementById('summary-data').textContent);
        const testResults = runData.results;
        
        let currentCategory = 'ALL';
        let currentStatus = 'ALL';
        
        // Initial setup
        document.addEventListener('DOMContentLoaded', () => {
            renderCharts();
            renderCategoryMetrics();
            renderTestCases();
            
            // Search Input handler
            document.getElementById('searchInput').addEventListener('input', () => {
                renderTestCases();
            });
        });

        function renderCharts() {
            const darkBg = 'rgba(0,0,0,0)';
            const gridColor = 'rgba(255,255,255,0.05)';
            const fontColor = '#94a3b8';

            // 1. Average Quality Scores Bar Chart
            const metricKeys = Object.keys(runData.metric_averages);
            const metricValues = Object.values(runData.metric_averages);

            const barTrace = {
                x: metricKeys.map(k => k.replace('_score', '').toUpperCase()),
                y: metricValues,
                type: 'bar',
                marker: {
                    color: 'rgba(99, 102, 241, 0.7)',
                    line: { color: '#6366f1', width: 1.5 }
                }
            };

            const barLayout = {
                paper_bgcolor: darkBg,
                plot_bgcolor: darkBg,
                xaxis: { gridcolor: gridColor, tickfont: {color: fontColor} },
                yaxis: { gridcolor: gridColor, range: [0, 1.05], tickfont: {color: fontColor} },
                margin: { t: 10, r: 10, b: 30, l: 30 },
                height: 280
            };

            Plotly.newPlot('metricsBarChart', [barTrace], barLayout, {responsive: true, displayModeBar: false});

            // 2. Latency vs Cost Scatter
            const scatterTrace = {
                x: testResults.map(r => r.latency),
                y: testResults.map(r => r.cost),
                mode: 'markers',
                type: 'scatter',
                text: testResults.map(r => r.test_case_id),
                marker: {
                    size: 10,
                    color: testResults.map(r => r.passed ? '#34d399' : '#f87171'),
                    line: { color: 'rgba(15,23,42,0.6)', width: 1 }
                }
            };

            const scatterLayout = {
                paper_bgcolor: darkBg,
                plot_bgcolor: darkBg,
                xaxis: { title: { text: 'Latency (s)', font: {color: fontColor} }, gridcolor: gridColor, tickfont: {color: fontColor} },
                yaxis: { title: { text: 'Cost ($)', font: {color: fontColor} }, gridcolor: gridColor, tickfont: {color: fontColor} },
                margin: { t: 10, r: 10, b: 40, l: 50 },
                height: 280
            };

            Plotly.newPlot('latencyCostScatter', [scatterTrace], scatterLayout, {responsive: true, displayModeBar: false});

            // 3. Status Donut Pie Chart
            const pieData = [{
                values: [runData.passed_count, runData.failed_count],
                labels: ['Passed', 'Failed'],
                hole: .6,
                type: 'pie',
                marker: {
                    colors: ['rgba(52, 211, 153, 0.7)', 'rgba(248, 113, 113, 0.7)']
                },
                textinfo: 'label+percent',
                showlegend: false
            }];

            const pieLayout = {
                paper_bgcolor: darkBg,
                plot_bgcolor: darkBg,
                margin: { t: 10, r: 10, b: 10, l: 10 },
                height: 220,
                textfont: {color: '#f8fafc'}
            };

            Plotly.newPlot('statusPieChart', pieData, pieLayout, {responsive: true, displayModeBar: false});
        }

        function renderCategoryMetrics() {
            // Group by category
            const categories = {};
            testResults.forEach(r => {
                const cat = r.category;
                if (!categories[cat]) {
                    categories[cat] = { count: 0, passed: 0, total_latency: 0, total_cost: 0 };
                }
                categories[cat].count++;
                if (r.passed) categories[cat].passed++;
                categories[cat].total_latency += r.latency;
                categories[cat].total_cost += r.cost;
            });

            const tbody = document.getElementById('categoryTableBody');
            tbody.innerHTML = '';
            
            Object.keys(categories).forEach(cat => {
                const data = categories[cat];
                const passRate = (data.passed / data.count) * 100;
                const avgLatency = data.total_latency / data.count;
                const avgCost = data.total_cost / data.count;
                
                const tr = document.createElement('tr');
                tr.className = 'border-b border-slate-900/40 hover:bg-slate-900/10 transition duration-150';
                tr.innerHTML = `
                    <td class="py-3 font-semibold text-slate-200 capitalize">${cat.replace('_', ' ')}</td>
                    <td class="py-3">${data.count}</td>
                    <td class="py-3"><span class="${passRate === 100 ? 'text-emerald-400' : passRate >= 70 ? 'text-amber-400' : 'text-rose-400'} font-semibold">${passRate.toFixed(0)}%</span></td>
                    <td class="py-3">${avgLatency.toFixed(2)}s</td>
                    <td class="py-3">$${avgCost.toFixed(5)}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function filterCategory(cat) {
            currentCategory = cat;
            // Update button styles
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => {
                btn.classList.remove('active', 'bg-indigo-500', 'text-white');
                btn.classList.add('bg-slate-900', 'text-slate-300', 'border-slate-800');
            });
            
            const activeBtn = Array.from(buttons).find(btn => btn.innerText.toLowerCase().includes(cat.toLowerCase().replace('_', ' ')));
            if (activeBtn) {
                activeBtn.classList.add('active', 'bg-indigo-500', 'text-white');
                activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
            } else if (cat === 'ALL') {
                buttons[0].classList.add('active', 'bg-indigo-500', 'text-white');
                buttons[0].classList.remove('bg-slate-900', 'text-slate-300');
            }
            
            renderTestCases();
        }

        function filterStatus() {
            currentStatus = document.getElementById('statusFilter').value;
            renderTestCases();
        }

        function toggleAccordion(id) {
            const panel = document.getElementById(`panel-${id}`);
            const arrow = document.getElementById(`arrow-${id}`);
            if (panel.classList.contains('hidden')) {
                panel.classList.remove('hidden');
                arrow.style.transform = 'rotate(180deg)';
            } else {
                panel.classList.add('hidden');
                arrow.style.transform = 'rotate(0deg)';
            }
        }

        function renderTestCases() {
            const container = document.getElementById('testCasesContainer');
            container.innerHTML = '';
            
            const searchQuery = document.getElementById('searchInput').value.toLowerCase();
            
            const filtered = testResults.filter(r => {
                const matchesSearch = r.test_case_id.toLowerCase().includes(searchQuery) || 
                                     r.prompt.toLowerCase().includes(searchQuery) ||
                                     r.response_text.toLowerCase().includes(searchQuery);
                const matchesCategory = currentCategory === 'ALL' || r.category === currentCategory;
                const matchesStatus = currentStatus === 'ALL' || 
                                     (currentStatus === 'PASSED' && r.passed) ||
                                     (currentStatus === 'FAILED' && !r.passed);
                return matchesSearch && matchesCategory && matchesStatus;
            });

            if (filtered.length === 0) {
                container.innerHTML = `
                    <div class="glass-card rounded-2xl p-8 text-center text-slate-500">
                        No test cases found matching filters.
                    </div>
                `;
                return;
            }

            filtered.forEach(r => {
                const card = document.createElement('div');
                card.className = `glass-card rounded-2xl overflow-hidden transition duration-300 border-l-4 ${r.passed ? 'border-l-emerald-500' : 'border-l-rose-500'}`;
                
                // Construct metrics badge list
                let metricsBadges = '';
                Object.keys(r.metrics).forEach(m => {
                    const res = r.metrics[m];
                    metricsBadges += `
                        <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${res.passed ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}">
                            ${m.replace('_score', '').toUpperCase()}: ${res.score}
                        </span>
                    `;
                });

                card.innerHTML = `
                    <!-- Header Bar -->
                    <div class="p-4 flex items-center justify-between cursor-pointer hover:bg-slate-900/20" onclick="toggleAccordion('${r.test_case_id}')">
                        <div class="flex items-center gap-4 flex-wrap">
                            <span class="font-mono font-bold text-indigo-400">${r.test_case_id}</span>
                            <span class="text-xs uppercase tracking-wider px-2 py-1 rounded bg-slate-900/80 text-slate-400 border border-slate-800">${r.category.replace('_', ' ')}</span>
                            <div class="flex gap-1.5">${metricsBadges}</div>
                        </div>
                        
                        <div class="flex items-center gap-4">
                            <span class="text-xs text-slate-500 font-mono">${r.latency.toFixed(2)}s | ${r.tokens} tokens</span>
                            <span class="px-2.5 py-1 text-xs font-semibold rounded-full ${r.passed ? 'bg-emerald-500/20 text-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.2)]' : 'bg-rose-500/20 text-rose-400 shadow-[0_0_8px_rgba(248,113,113,0.2)]'}">
                                ${r.passed ? 'Passed' : 'Failed'}
                            </span>
                            <svg id="arrow-${r.test_case_id}" class="h-5 w-5 text-slate-500 transition-transform duration-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                            </svg>
                        </div>
                    </div>

                    <!-- Details Accordion Panel -->
                    <div id="panel-${r.test_case_id}" class="hidden border-t border-slate-900/60 p-6 bg-slate-950/20 space-y-6">
                        
                        <!-- Details grid -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <!-- Left: Inputs -->
                            <div class="space-y-4">
                                <div>
                                    <h5 class="text-xs uppercase font-semibold text-slate-500 tracking-wider">Prompt</h5>
                                    <div class="mt-1.5 p-3 rounded-xl bg-slate-950/80 font-mono text-sm text-slate-200 border border-slate-900 custom-scrollbar max-h-48 overflow-y-auto whitespace-pre-wrap">${r.prompt}</div>
                                </div>
                                ${r.context ? `
                                <div>
                                    <h5 class="text-xs uppercase font-semibold text-slate-500 tracking-wider">Context</h5>
                                    <div class="mt-1.5 p-3 rounded-xl bg-slate-950/80 font-mono text-sm text-slate-300 border border-slate-900 custom-scrollbar max-h-48 overflow-y-auto whitespace-pre-wrap">${r.context}</div>
                                </div>` : ''}
                            </div>
                            
                            <!-- Right: Output -->
                            <div class="space-y-4">
                                <div>
                                    <h5 class="text-xs uppercase font-semibold text-slate-500 tracking-wider">Response</h5>
                                    <div class="mt-1.5 p-3 rounded-xl bg-slate-950/80 font-mono text-sm text-slate-100 border border-slate-900 custom-scrollbar max-h-48 overflow-y-auto whitespace-pre-wrap">${r.response_text}</div>
                                </div>
                                ${r.expected_response ? `
                                <div>
                                    <h5 class="text-xs uppercase font-semibold text-slate-500 tracking-wider">Ground Truth</h5>
                                    <div class="mt-1.5 p-3 rounded-xl bg-slate-950/80 font-mono text-sm text-slate-300 border border-slate-900 custom-scrollbar max-h-48 overflow-y-auto whitespace-pre-wrap">${r.expected_response}</div>
                                </div>` : ''}
                            </div>
                        </div>

                        <!-- Metrics Scoring breakdown details -->
                        <div class="mt-6">
                            <h5 class="text-xs uppercase font-semibold text-slate-500 tracking-wider mb-2">Metrics Detailed Assessment</h5>
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                ${Object.keys(r.metrics).map(m => {
                                    const m_res = r.metrics[m];
                                    return `
                                        <div class="p-4 rounded-xl bg-slate-900/40 border border-slate-900 flex flex-col justify-between">
                                            <div>
                                                <span class="text-xs font-semibold text-slate-400 capitalize">${m.replace('_score', '').replace('_', ' ')}</span>
                                                <h4 class="text-xl font-bold mt-1 ${m_res.passed ? 'text-emerald-400' : 'text-rose-400'}">${m_res.score}</h4>
                                                <p class="text-[10px] text-slate-500 mt-1">Threshold: >= ${m_res.threshold}</p>
                                            </div>
                                            <div class="mt-3 pt-2 border-t border-slate-800 text-xs font-mono text-slate-400 break-words max-h-32 overflow-y-auto">
                                                ${JSON.stringify(m_res.details, null, 2).replace(/\\n/g, '<br>')}
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>

                    </div>
                `;
                container.appendChild(card);
            });
        }
    </script>
</body>
</html>
"""

def generate_html_report(summary: EvaluationRunSummary, output_path: str) -> str:
    """
    Renders the rich executive HTML dashboard template using Jinja2 and writes to disk.
    """
    path = Path(output_path)
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"report_{summary.run_id}.html"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert run summary to JSON string for injected Javascript consumption
        summary_json = summary.model_dump_json()

        template = Template(HTML_TEMPLATE)
        rendered = template.render(
            summary=summary,
            summary_json=summary_json
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(rendered)

        logger.info(f"Premium HTML report generated successfully at {path}")
        return str(path)
    except Exception as e:
        logger.error(f"Failed to generate HTML report at {path}: {str(e)}")
        raise e
