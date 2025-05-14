
import { useState } from "react";
import { BuildingAccessForm } from "@/components/BuildingAccessForm";
import { AccessibilityResult, AccessibilityData } from "@/components/AccessibilityResult";
import { fetchBuildingAccessibility } from "@/utils/mockData";
import { useToast } from "@/components/ui/use-toast";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [accessibilityData, setAccessibilityData] = useState<AccessibilityData | null>(null);
  const { toast } = useToast();

  const handleBuildingSubmit = async (buildingId: string) => {
    setIsLoading(true);
    try {
      const data = await fetchBuildingAccessibility(buildingId);
      setAccessibilityData(data);
    } catch (error) {
      console.error("Failed to fetch building data:", error);
      toast({
        title: "Error",
        description: "Failed to load building accessibility data. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-sky-50 to-white">
      <div className="container mx-auto px-4 py-12">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-primary mb-4">
            Transport Connectivity Score(TCS)
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Discover how accessible buildings are using public transportation
          </p>
        </header>

        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8 items-start">
            <div className="md:top-24">
              <div className="space-y-6">
                <div className="bg-white p-6 rounded-xl shadow-sm border">
                  <h2 className="text-2xl font-semibold mb-4">How It Works</h2>
                  <ol className="space-y-3">
                    <li className="flex gap-2">
                      <span className="flex-shrink-0 h-6 w-6 rounded-full bg-primary text-white flex items-center justify-center text-sm">1</span>
                      <span>Select a building from the dropdown menu</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="flex-shrink-0 h-6 w-6 rounded-full bg-primary text-white flex items-center justify-center text-sm">2</span>
                      <span>Click the button to check accessibility</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="flex-shrink-0 h-6 w-6 rounded-full bg-primary text-white flex items-center justify-center text-sm">3</span>
                      <span>View TCS and detailed explanation</span>
                    </li>
                  </ol>
                </div>

                <BuildingAccessForm onSubmit={handleBuildingSubmit} isLoading={isLoading} />
              </div>
            </div>

            <div className="space-y-6">
              {accessibilityData ? (
                <AccessibilityResult data={accessibilityData} />
              ) : (
                <div className="bg-white p-8 rounded-xl shadow-sm border text-center">
                  <div className="text-6xl mb-4">🚍</div>
                  <h3 className="text-xl font-medium mb-2">
                    Select a building to see its accessibility score
                  </h3>
                  <p className="text-muted-foreground">
                    We will calculate the Transport Connectivity Score for the building you select and explain what it means for you
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        <footer className="mt-24 text-center text-sm text-muted-foreground">
          <p>Transport Connnectivity Score helps you make informed decisions using transport accessibility data</p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
