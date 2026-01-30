//Mock Database with History Logs
const registeredUsers = [
    { 
        id: "R-101", name: "Rahul Sharma", type: "Resident", status: "Active",
        history: ["2026-01-30 10:00 AM", "2026-01-29 06:30 PM", "2026-01-28 08:15 AM", "2026-01-27 09:00 PM"]
    },
    { 
        id: "R-102", name: "Priya Das", type: "Resident", status: "Active",
        history: ["2026-01-29 11:00 AM", "2026-01-25 04:20 PM"]
    },
    { 
        id: "S-205", name: "Amit Verma", type: "Staff", status: "Active",
        history: ["2026-01-30 08:30 AM", "2026-01-29 08:35 AM", "2026-01-28 08:30 AM", "2026-01-27 08:32 AM"]
    },
    { 
        id: "V-901", name: "Delivery #44", type: "Visitor", status: "Pending",
        history: ["2026-01-30 02:45 PM"]
    }
];

// --- PREMIUM LOGIN FORM HANDLER (from index1) ---
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById("loginForm");
    if (form) {
        form.addEventListener("submit", function(e){
            e.preventDefault();

            const guardId = document.getElementById("guardId").value.trim();
            const guardPassword = document.getElementById("guardPassword").value.trim();
            const msg = document.getElementById("premiumLoginMsg");

            if(guardId === "" || guardPassword === ""){
                msg.style.color = "#ffbaba";
                msg.textContent = "Verification failed: Missing credentials.";
                return;
            }

            if(guardPassword.length < 4){
                msg.style.color = "#ffbaba";
                msg.textContent = "Low trust credentials detected. Access denied.";
                return;
            }

            msg.style.color = "#baffc9";
            msg.textContent = "Identity verified. Access granted ✔";
            
            // Navigate to dashboard after successful verification
            setTimeout(() => {
                navTo('dashboardHome');
            }, 1500);
        });
    }
});

// Premium password toggle function
function premiumTogglePassword(){
    const pwd = document.getElementById("guardPassword");
    pwd.type = pwd.type === "password" ? "text" : "password";
}

// --- NAVIGATION ---
function navTo(pageId) {
    // Hide all main pages
    ['loginPage', 'dashboardHome', 'databasePage'].forEach(id => {
        document.getElementById(id).classList.add('hidden');
    });

    // Show target page
    document.getElementById(pageId).classList.remove('hidden');

    // If opening database, render the table
    if (pageId === 'databasePage') renderTable();
}

// --- TABLE LOGIC ---
function renderTable() {
    const tbody = document.getElementById('userTableBody');
    tbody.innerHTML = '';
    
    registeredUsers.forEach(user => {
        const lastAccess = user.history[0] || "N/A";
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-family:'Courier New'">${user.id}</td>
            <td style="font-weight:bold;">${user.name}</td>
            <td>${user.type}</td>
            <td style="color:#94a3b8">${lastAccess}</td>
            <td><span class="status-badge">${user.status}</span></td>
        `;
        // Add click event to open User Modal
        row.onclick = () => openUserModal(user);
        tbody.appendChild(row);
    });
}

// --- USER MODAL LOGIC (New Feature) ---
function openUserModal(user) {
    const modal = document.getElementById('userModal');
    
    // Fill data
    document.getElementById('modalUserName').innerText = user.name;
    document.getElementById('modalUserId').innerText = user.id;
    document.getElementById('modalEntryCount').innerText = user.history.length;
    
    // Fill history list
    const historyList = document.getElementById('modalHistoryList');
    historyList.innerHTML = ''; 
    user.history.forEach(time => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `<span>ENTRY ALLOWED</span> <span>${time}</span>`;
        historyList.appendChild(div);
    });

    modal.classList.remove('hidden');
}

function closeUserModal() {
    document.getElementById('userModal').classList.add('hidden');
}

// --- SCANNER LOGIC ---
async function openScanner() {
    const modal = document.getElementById('scannerModal');
    modal.classList.remove('hidden');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('webcam');
        video.srcObject = stream;
        startMockScan(); 
    } catch (err) {
        alert("Camera permission denied! Please allow access.");
    }
}

function closeScanner() {
    const modal = document.getElementById('scannerModal');
    modal.classList.add('hidden');

    const video = document.getElementById('webcam');
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    document.getElementById('scanOverlay').className = "scan-overlay scanning";
    const resultText = document.getElementById('scanResult');
    resultText.innerText = "ALIGN FACE...";
    resultText.style.color = "white";
}

function startMockScan() {
    const resultText = document.getElementById('scanResult');
    const overlay = document.getElementById('scanOverlay');
    
    resultText.innerText = "CHECKING LIVENESS...";
    
    setTimeout(() => {
        overlay.className = "scan-overlay success";
        resultText.innerHTML = "✅ VERIFIED: RAHUL SHARMA";
        resultText.style.color = "#22c55e"; 
    }, 3000);
}