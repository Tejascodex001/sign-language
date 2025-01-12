document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('start-camera').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/start-camera', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Camera started:', data);
            document.getElementById('video-feed').style.display = 'block';
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    document.getElementById('stop-camera').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/stop-camera', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Camera stopped:', data);
            document.getElementById('video-feed').style.display = 'none';
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    document.getElementById('start-detection').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/start-detection', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Detection started:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    document.getElementById('stop-detection').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/stop-detection', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Detection stopped:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    document.getElementById('save-text').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/save-text', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Text saved:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    document.getElementById('convert-to-audio').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/convert-to-audio', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Text converted to audio:', data);
            document.getElementById('audio-source').src = 'http://127.0.0.1:8000/get-audio?' + new Date().getTime();
            document.getElementById('audio-player').load();
            document.getElementById('audio-player').play();
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    document.getElementById('clear-text').addEventListener('click', function() {
        fetch('http://127.0.0.1:8000/clear-text', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Text cleared:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    });

    // Add event listener for spacebar key press
    document.addEventListener('keydown', function(event) {
        if (event.code === 'Space') {
            fetch('http://127.0.0.1:8000/add-space', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                console.log('Space added to detected letters:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }
    });
});
