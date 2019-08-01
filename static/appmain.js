function tableinit() {
	d3.json("/summary_edata", function(details) {
		console.log(details);
		d3.select("tbody")
	  .selectAll("tr")
	  .data(details)
	  .enter()
	  .append("tr")
	  .html(function(d) {
	    return `<td>${d.Country}</td><td>${d.MS17}</td><td>${d.MS18}</td><td>${d.MSChange}</td><td>${d.Value_yr17}</td><td>${d.Value_yr18}</td>`;
	  });
    });
}

tableinit();

function top10veginit() {
	d3.json("/top10veg_edata", function(vegexport) {
		console.log(vegexport);
		var vegtableInit = d3.select(".table1").select("tbody");
		var veggieStart = vegtableInit.selectAll("tr").data(vegexport);
		veggieStart.enter()
		  .append("tr")
		  .merge(veggieStart)
		  .html(function(d) {
			return `<td>${d.Product}</td><td>${d.Average}</td><td>${d.Country}</td>`;
		  });

	});
	
}

top10veginit();

function productFlow(product) {
	if (product == "Fruit") {
		d3.json("/top10fruit_edata", function(fruitExport) {
			console.log(fruitExport);
			var fruitCheckInit = d3.select(".table1")
						   .select("tbody")
						   .selectAll("tr")
						   .data(fruitExport);
			fruitCheckInit.enter()
						   .append("tr")
						   .merge(fruitCheckInit)
						   .html(function(d) {
							   return `<td>${d.Product}</td><td>${d.Average}</td><td>${d.Country}</td>`;
						   });
			fruitCheckInit.exit().remove();
		});	    
	}
	else {
		d3.json("/top10veg_edata", function(vegExport) {
            console.log(vegExport);
			var vegCheckInit = d3.select(".table1")
						   .select("tbody")
						   .selectAll("tr")
						   .data(vegExport)
			vegCheckInit.enter()
						.append("tr")
						.merge(vegCheckInit)
						.html(function(d) {
							return `<td>${d.Product}</td><td>${d.Average}</td><td>${d.Country}</td>`;
						});
		    vegCheckInit.exit().remove();
		});
	}
}

// Initial Bar Chart
function top10VegExpInitChart() {
	d3.json("/top10veg_edata_bar", function(vegData) {
		var vegExportAvg = vegData.Average;
		var vegExportOverall = vegData.OverallExport;
		console.log(vegExportAvg);
		console.log(vegExportOverall);

		var trace1 = {
			x: vegExportOverall,
			y: vegExportAvg,
			type: "bar"
		};

		var barData = [trace1];

		var layout1 = {
			title: "Export Veggie Average by Veggie",
		  };
		  
		Plotly.newPlot("plot1", barData, layout1);

		var trace2 = {
			labels: vegExportOverall,
			values: vegExportAvg,
            type: "pie"
		}
		
		var pieData = [trace2];
		var layout2 = {
			title: "Export Veggie Percentage",
		};

        Plotly.newPlot("plot2", pieData, layout2)

	});
}

top10VegExpInitChart();

function getChart(productType) {
	if (productType == "Fruit") {
		d3.json("/top10fruit_edata_bar", function(fruitData) {
			var fruitAvg = fruitData.Average;
			var fruitList = fruitData.OverallExport;

			var trace = {
				x: fruitList,
				y: fruitAvg,
				type: 'bar'
			};

			var fruitExp = [trace];
			var layout = {
				title: "Export Fruit Average by Fruit",
			};
			Plotly.newPlot("plot1", fruitExp, layout);

			var traceFruit = {
				labels: fruitList,
				values: fruitAvg,
				type: 'pie'
			}
			var FruitExpPie = [traceFruit];
			var layout3 = {
				title: "Export Fruit Percentage",
			}
            Plotly.newPlot("plot2", FruitExpPie, layout3);

		});
	}
	else {
		d3.json("/top10veg_edata_bar", function(vegData) {
			var vegExportAvg = vegData.Average;
			var vegExportOverall = vegData.OverallExport;

			var trace = {
				x: vegExportOverall,
				y: vegExportAvg,
				type: "bar"
			};

            var exportData = [trace];

		    var layout = {
			    title: "Export Veggie Average by Veggie",
		    };
		  
			Plotly.newPlot("plot1", exportData, layout);
			
			var traceVeg = {
				labels: vegExportOverall,
				values: vegExportAvg,
				type: "pie"
			};
			var exportPie = [traceVeg];
			var layoutVeg = {
				title: "Export Veggie Percentage",
			};

			Plotly.newPlot("plot2", exportPie, layoutVeg);
		});
	}
}
