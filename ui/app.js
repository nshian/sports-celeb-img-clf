Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {        
        var url = "http://127.0.0.1:8080/classify_image";

        $.post(url, {
            image_data: file.dataURL
        },function(data) {
            console.log(data);
            if (!data || data.length==0) {
                $("#resultHolder").hide();
                $("#error").show();
                return;
            }
            for (let i = 0; i < data.length; i++) {
                pred = data[i];
                $("#error").hide();
                $("#resultHolder").show();
                $("#resultHolder").html($(`[data-player="${pred}"`).html());
            }
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    console.log("Ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    init();
});