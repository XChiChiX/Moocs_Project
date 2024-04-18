let startBtn = document.querySelector(".start-btn");
const downloadBtns = document.querySelectorAll('.download-btn');

startBtn.addEventListener("click", () => {
    let startConfirm = confirm("確定要啟動ㄇ");
    if(startConfirm){
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
                .then(data => {
                    console.log(data)
                    downloadBtns.forEach(downloadBtn => {
                        downloadBtn.style.display = "inline-block";
                    })
                })
                .catch(error => console.error('Error:', error));
        } else {
            console.error('未選擇文件');
        }
    }
});

downloadBtns.forEach(downloadBtn => {
    downloadBtn.style.display = "none";
    downloadBtn.addEventListener("click", () => {
        // if(downloadBtn.id == "DL-btn1"){
        //     fileName = "textOnly.txt";
        // }
        // else {
        //     fileName = "textAndTime.txt";
        // }
        fileName = "result.zip";
        fetch(`/api/download/${fileName}`, {
            method: 'POST',
        })
            .then(response => response.blob())
            .then(blob => {
                // 創建一個 URL 對象
                const url = window.URL.createObjectURL(blob);
                // 創建一個 a 元素，將 URL 賦值給其 href 屬性，並觸發點擊
                const a = document.createElement('a');
                a.href = url;
                a.download = fileName;
                a.click();
                // 釋放 URL 對象
                window.URL.revokeObjectURL(url);
            })
            .catch(error => console.error('Error:', error));
    })
})