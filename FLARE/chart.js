export function initCharts() {
    // ========= Water level =========
    const ctxWater = document.getElementById('chartWater');
    new Chart(ctxWater, {
        type: 'line',
        data: {
            labels: generateLast24hLabels(), // ['01:00', '02:00', ..., '24:00']
            datasets: [
                {
                    label: 'Station 1',
                    data: mockWater(24, 10, 20),
                    borderColor: '#3498db',
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Station 2',
                    data: mockWater(24, 5, 18),
                    borderColor: '#e67e22',
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Station 3',
                    data: mockWater(24, 12, 22),
                    borderColor: '#2ecc71',
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Station 4',
                    data: mockWater(24, 8, 15),
                    borderColor: '#e74c3c',
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of reports / hour'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time (hours)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
    // const ctxWater = document.getElementById('chartWater');
    // new Chart(ctxWater, {
    //     type: 'bar',
    //     data: {
    //         labels: ['Station 1', 'Station 2', 'Station 3', 'Station 4'],
    //         datasets: [{
    //             label: 'Total number of reports',
    //             data: [12, 8, 14, 6], // ⚠️ giả lập – thay bằng số thực tế nếu có
    //             backgroundColor: '#3498db'
    //         }]
    //     },
    //     options: {
    //         responsive: true,
    //         maintainAspectRatio: false,
    //         scales: {
    //             y: {
    //                 beginAtZero: true,
    //                 title: {
    //                     display: true,
    //                     text: 'Evaluation report'
    //                 }
    //             }
    //         },
    //         plugins: {
    //             legend: {
    //                 display: false
    //             }
    //         }
    //     }
    // });
    // ========= Device status =========

    // ========= Zone distribution =========

    const ctxZone = document.getElementById('chartZone1');
    new Chart(ctxZone, {
        type: 'doughnut',
        data: {
            labels: ['Red zone', 'Green zone'],
            datasets: [{
                data: [3, 2],
                backgroundColor: ['#fd2b2b', '#35f7ac'],
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            cutout: '60%'
        }
    });

    const ctxAlert = document.getElementById('chartAlertType');
    new Chart(ctxAlert, {
        type: 'doughnut',
        data: {
            labels: ['Camera', 'Drone', 'Pi'],
            datasets: [{
                data: [2, 3, 1],
                backgroundColor: ['#f39c12', '#e74c3c', '#3498db'],
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            cutout: '60%'
        }
    });


}
export function generateLast24hLabels() {
    const arr = [];
    const now = new Date();
    for (let i = 23; i >= 0; i--) {
        const d = new Date(now.getTime() - i * 3600000);
        arr.push(d.getHours().toString().padStart(2, '0') + ':00');
    }
    return arr;
}
export function mockWater(n, min, max) {
    return [...Array(n)].map(() => Math.floor(Math.random() * (max - min) + min));
}