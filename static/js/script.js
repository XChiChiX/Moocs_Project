let startBtn = document.querySelector(".start-btn");

startBtn.addEventListener("click", () => {
    let file = document.getElementById("original-video").files[0];
    console.log(file)

    if (file) {
        const formData = new FormData();
        formData.append('file', file);
    
        // 使用fetch方法將資料傳到後端
        fetch('/api/upload', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error('Error:', error));
        } else {
            console.error('未選擇文件');
        }
});
