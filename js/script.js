// Document ready event
document.addEventListener('DOMContentLoaded', function() {
    console.log('Flex Exercise Monitoring System documentation loaded');
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('nav a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            window.scrollTo({
                top: targetElement.offsetTop - 20,
                behavior: 'smooth'
            });
        });
    });
    
    // Image loading verification
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('error', function() {
            console.error(`Failed to load image: ${this.src}`);
            this.src = 'images/placeholder.jpg';
            this.alt = 'Image not available';
        });
    });
    
    // Dynamically load code files with collapsible sections
    displayCodeFiles('code-file-list');
    
    // Dynamically load data files with collapsible sections
    displayDataFiles('data-file-list');
    
    // Add dynamic gallery zoom functionality
    setupGalleryZoom();
});

// Function to display code files with collapsible sections
function displayCodeFiles(targetElementId) {
    const targetElement = document.getElementById(targetElementId);
    
    // Your actual code files from the project
    const codeFiles = [
        'Exercise_classification_MLP.py',
        'Exercise_classification_OtherModels.py',
        'rep_count.py',
        'cnn-lstm_rep_count.py',
    ];
    
    // Clear loading message
    targetElement.innerHTML = '';
    
    // Create collapsible section
    const collapsibleDiv = document.createElement('div');
    collapsibleDiv.className = 'collapsible-section';
    
    // Create header button
    const headerButton = document.createElement('button');
    headerButton.className = 'collapsible-header';
    headerButton.innerHTML = 'Code Files <span class="toggle-icon">▼</span>';
    collapsibleDiv.appendChild(headerButton);
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'collapsible-content';
    
    // Create table structure
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    // Add headers
    ['File Name', 'File Type', 'Download'].forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    const tbody = document.createElement('tbody');
    
    // Add each file to the table
    codeFiles.forEach(filename => {
        const row = document.createElement('tr');
        
        // File name cell
        const nameCell = document.createElement('td');
        nameCell.textContent = filename;
        row.appendChild(nameCell);
        
        // File type cell
        const typeCell = document.createElement('td');
        const extension = filename.split('.').pop().toLowerCase();
        typeCell.textContent = getFileTypeDescription(extension);
        row.appendChild(typeCell);
        
        // Download cell
        const downloadCell = document.createElement('td');
        const downloadLink = document.createElement('a');
        
        // Determine the correct path based on file extension
        let filePath = '';
        if (extension === 'css') {
            filePath = `css/${filename}`;
        } else if (extension === 'js') {
            filePath = `js/${filename}`;
        } else if (extension === 'html') {
            filePath = filename;
        } else if (extension === 'md') {
            filePath = filename;
        } else if (extension === 'txt') {
            filePath = filename;
        } else {
            filePath = `code/${filename}`;
        }
        
        downloadLink.href = filePath;
        downloadLink.className = 'download-btn';
        downloadLink.setAttribute('download', '');
        downloadLink.textContent = 'Download';
        downloadCell.appendChild(downloadLink);
        row.appendChild(downloadCell);
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    contentDiv.appendChild(table);
    collapsibleDiv.appendChild(contentDiv);
    targetElement.appendChild(collapsibleDiv);
    
    // Add event listener for collapsible functionality
    headerButton.addEventListener('click', function() {
        this.classList.toggle('active');
        const content = this.nextElementSibling;
        const toggleIcon = this.querySelector('.toggle-icon');
        
        if (content.style.maxHeight) {
            content.style.maxHeight = null;
            toggleIcon.textContent = '▼';
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
            toggleIcon.textContent = '▲';
        }
    });
    
    // Initially expand the section
    headerButton.click();
}

// Function to display data files with collapsible sections
function displayDataFiles(targetElementId) {
    const targetElement = document.getElementById(targetElementId);
    
    // Clear loading message
    targetElement.innerHTML = '';
    
    // Your actual data folders and files from the project
    const dataFolders = [
        {
            name: 'EMG Bicep',
            folderPath: 'EMG Bicep', 
            files: [
                'Bicep_Count_1.csv',
                'Bicep_Count_2.csv',
                'Bicep_Count_3.csv',
                'Bicep_Count_4.csv',
                'Bicep_Count_5.csv',
                'Bicep_Count_6.csv',
                'Bicep_Count_7.csv',
                'Bicep_Count_8.csv',
                'Bicep_Count_9.csv'
            ]
        },
        {
            name: 'EMG Shoulder',
            folderPath: 'EMG Shoulder',
            files: [
                'Shoulder_1.csv',
                'Shoulder_2.csv',
                'Shoulder_3.csv',
                'Shoulder_4.csv',
                'Shoulder_5.csv',
                'Shoulder_6.csv',
                'Shoulder_7.csv',
                'Shoulder_8.csv',
                'Shoulder_9.csv',
                'Shoulder_10.csv',
                'Shoulder_11.csv'
            ]
        },
        {
            name: 'IMU Bicep',
            folderPath: 'IMU Bicep',
            files: [
                'NewBicep_1.csv',
                'NewBicep_2.csv',
                'NewBicep_3.csv',
                'NewBicep_4.csv',
                'NewBicep_5.csv',
                'NewBicep_6.csv',
                'NewBicep_7.csv'
            ]
        },
        {
            name: 'IMU Shoulder',
            folderPath: 'IMU Shoulder',
            files: [
                'NewPress_1.csv',
                'NewPress_2.csv',
                'NewPress_3.csv',
                'NewPress_4.csv',
                'NewPress_5.csv',
                'NewPress_6.csv',
                'NewPress_7.csv',
                'NewShoulder_1.csv',
                'NewShoulder_2.csv',
                'NewShoulder_2_2.csv',
                'NewShoulder_3.csv',
                'NewShoulder_4.csv',
                'NewShoulder_5.csv',
                'NewShoulder_6.csv',
                'NewShoulder_7.csv'
            ]
        }
    ];
    
    // Create main collapsible section
    const mainCollapsibleDiv = document.createElement('div');
    mainCollapsibleDiv.className = 'collapsible-section';
    
    // Create main header button
    const mainHeaderButton = document.createElement('button');
    mainHeaderButton.className = 'collapsible-header';
    mainHeaderButton.innerHTML = 'All Data Files <span class="toggle-icon">▼</span>';
    mainCollapsibleDiv.appendChild(mainHeaderButton);
    
    // Create main content container
    const mainContentDiv = document.createElement('div');
    mainContentDiv.className = 'collapsible-content';
    
    // Process each data folder
    dataFolders.forEach(folder => {
        // Create folder collapsible section
        const folderDiv = document.createElement('div');
        folderDiv.className = 'collapsible-section nested';
        
        // Create folder header button
        const folderButton = document.createElement('button');
        folderButton.className = 'collapsible-header folder';
        folderButton.innerHTML = `${folder.name} <span class="toggle-icon">▼</span>`;
        folderDiv.appendChild(folderButton);
        
        // Create folder content container
        const folderContent = document.createElement('div');
        folderContent.className = 'collapsible-content';
        
        // Create table for this folder
        const table = document.createElement('table');
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        // Add headers
        ['File Name', 'File Type', 'Download'].forEach(headerText => {
            const th = document.createElement('th');
            th.textContent = headerText;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        const tbody = document.createElement('tbody');
        
        // Add files in this folder
        folder.files.forEach(filename => {
            const row = document.createElement('tr');
            
            // File name cell
            const nameCell = document.createElement('td');
            nameCell.textContent = filename;
            row.appendChild(nameCell);
            
            // File type cell
            const typeCell = document.createElement('td');
            const extension = filename.split('.').pop().toLowerCase();
            typeCell.textContent = getFileTypeDescription(extension);
            row.appendChild(typeCell);
            
            // Download cell
            const downloadCell = document.createElement('td');
            const downloadLink = document.createElement('a');
            // Use folder path with spaces encoded for URLs
            const encodedFolderPath = encodeURIComponent(folder.folderPath);
            downloadLink.href = `data/${encodedFolderPath}/${filename}`;
            downloadLink.className = 'download-btn';
            downloadLink.setAttribute('download', '');
            downloadLink.textContent = 'Download';
            downloadCell.appendChild(downloadLink);
            row.appendChild(downloadCell);
            
            tbody.appendChild(row);
        });
        
        table.appendChild(tbody);
        folderContent.appendChild(table);
        folderDiv.appendChild(folderContent);
        mainContentDiv.appendChild(folderDiv);
        
        // Add event listener for folder collapsible functionality
        folderButton.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent bubbling to parent collapsible
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            const toggleIcon = this.querySelector('.toggle-icon');
            
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
                toggleIcon.textContent = '▼';
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
                toggleIcon.textContent = '▲';
                
                // Update parent's max height to accommodate this expansion
                const parentContent = this.parentElement.parentElement;
                if (parentContent.style.maxHeight) {
                    parentContent.style.maxHeight = parentContent.scrollHeight + content.scrollHeight + "px";
                }
            }
        });
    });
    
    mainCollapsibleDiv.appendChild(mainContentDiv);
    targetElement.appendChild(mainCollapsibleDiv);
    
    // Add event listener for main collapsible functionality
    mainHeaderButton.addEventListener('click', function() {
        this.classList.toggle('active');
        const content = this.nextElementSibling;
        const toggleIcon = this.querySelector('.toggle-icon');
        
        if (content.style.maxHeight) {
            content.style.maxHeight = null;
            toggleIcon.textContent = '▼';
        } else {
            content.style.maxHeight = content.scrollHeight + "px";
            toggleIcon.textContent = '▲';
        }
    });
    
    // Initially expand the main section
    mainHeaderButton.click();
}

// Function to get descriptive file type based on extension
function getFileTypeDescription(extension) {
    const typeMap = {
        'py': 'Python Source Code',
        'js': 'JavaScript Source Code',
        'sql': 'SQL Database Script',
        'cpp': 'C++ Source Code',
        'h': 'C/C++ Header File',
        'json': 'JSON Configuration File',
        'txt': 'Plain Text',
        'md': 'Markdown',
        'csv': 'CSV Document',
        'xlsx': 'Excel Spreadsheet',
        'pdf': 'PDF Document',
        'html': 'HTML Document',
        'css': 'CSS Source'
    };
    
    return typeMap[extension] || `${extension.toUpperCase()} File`;
}

// Function to set up gallery zoom functionality
function setupGalleryZoom() {
    const galleryItems = document.querySelectorAll('.gallery-item img');
    
    galleryItems.forEach(item => {
        item.addEventListener('click', function() {
            // Create modal for image zoom
            const modal = document.createElement('div');
            modal.classList.add('image-modal');
            
            const modalImg = document.createElement('img');
            modalImg.src = this.src;
            
            const closeBtn = document.createElement('span');
            closeBtn.classList.add('close-modal');
            closeBtn.innerHTML = '&times;';
            closeBtn.addEventListener('click', function() {
                modal.remove();
            });
            
            modal.appendChild(closeBtn);
            modal.appendChild(modalImg);
            document.body.appendChild(modal);
            
            // Close modal when clicking outside the image
            modal.addEventListener('click', function(e) {
                if (e.target === modal) {
                    modal.remove();
                }
            });
        });
    });
    
    // Add modal styles dynamically
    const style = document.createElement('style');
    style.textContent = `
        .image-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .image-modal img {
            max-width: 90%;
            max-height: 90%;
            object-fit: contain;
        }
        
        .close-modal {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .subfolder-header {
            background-color: #f0f8ff;
            font-weight: bold;
            padding: 10px;
            border-left: 4px solid #3498db;
        }
    `;
    document.head.appendChild(style);
}

