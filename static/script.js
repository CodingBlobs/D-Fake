const post_url = `${window.location.href}/api/estimate`

async function upload_image(image) {
    console.log("Upload and classify called!");
    const data = await fetch('api/estimate', {
        method: 'POST',
        body: JSON.stringify({"image_b64": image}),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(async (response) => {
        const res = await response.json();
        return res;
    }).catch((err) => {
        console.log(err);
    });
    const res_b64 = data['classified_b64'];
    const estimated_price = parseFloat(data['price']).toFixed(4);
    const innerHTML = `<p class="mt-ne3"><h2> Result: </h2></p>
                <img src="data:image/png;base64,${res_b64}" alt="Classified Image">
                <p><h5 class="text-success"> Estimated Price: ${estimated_price} ETH </h3><p>`;
    document.getElementById('predictedImage').innerHTML = innerHTML;
    document.getElementById('hasUploaded1').remove();
    document.getElementById('hasUploaded2').remove();
    $('#loadingModal').modal("hide");
    if (estimated_price === -1) {
        window.alert("Error: please contact admin and report this bug!")
    }
}