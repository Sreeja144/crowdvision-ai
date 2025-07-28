import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Eye, Moon, Sun, Upload, Play, Mail, CheckCircle, XCircle, Info, Users, UserX, AlertTriangle, Loader2, Shield, Activity, Camera, Square } from 'lucide-react';

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [selectedSource, setSelectedSource] = useState('default');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [emailForReport, setEmailForReport] = useState(''); // This email will be sent to backend on /start
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [monitoringStatus, setMonitoringStatus] = useState('Ready to start monitoring.');
  const [monitoringResults, setMonitoringResults] = useState({
    current_crowd_count: 0,
    maximum_crowd_count: 0,
    standing_detections: 0,
    bending_detections: 0,
    unknown_faces: 0,
    known_faces: 0,
    theft_detected: false
  });
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [videoFeedKey, setVideoFeedKey] = useState(Date.now()); // Key to force re-render of img for video feed

  // Real-time updates when monitoring is active
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isMonitoring) {
      const fetchStatus = async () => {
        try {
        
          const response = await axios.get("https://bd8b46660df3.ngrok-free.app/status");

          setMonitoringResults(response.data);
        } catch (error) {
          console.error("Error fetching status:", error);
          // If status fails while monitoring, assume backend stopped or crashed
          setErrorMessage("Failed to fetch real-time status. Monitoring may have stopped.");
          setIsMonitoring(false); // Stop frontend monitoring state
          setMonitoringStatus('Monitoring interrupted.');
          setVideoFeedKey(Date.now()); // Refresh to clear potentially stale image
        }
      };

      fetchStatus(); // Initial fetch
      interval = setInterval(fetchStatus, 2000); // Update every 2 seconds
    }

    return () => clearInterval(interval);
  }, [isMonitoring]);

  // Cleanup video preview URL on component unmount
  useEffect(() => {
    return () => {
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    };
  }, [videoPreviewUrl]);

  const handleSourceChange = (source: string) => {
    setSelectedSource(source);
    setUploadedFile(null);
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(null);
    resetMonitoringResults();
    setErrorMessage(null);
    setMonitoringStatus('Ready to start monitoring.');
    setVideoFeedKey(Date.now()); // Reset video feed display
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files?.[0]) {
      const file = event.target.files[0];
      setUploadedFile(file);
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      setVideoPreviewUrl(URL.createObjectURL(file));
    } else {
      setUploadedFile(null);
      if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
      setVideoPreviewUrl(null);
    }
    resetMonitoringResults();
    setErrorMessage(null);
    setMonitoringStatus('Ready to start monitoring.');
    setVideoFeedKey(Date.now()); // Reset video feed display
  };

  const resetMonitoringResults = () => {
    setMonitoringResults({
      current_crowd_count: 0,
      maximum_crowd_count: 0,
      standing_detections: 0,
      bending_detections: 0,
      unknown_faces: 0,
      known_faces: 0,
      theft_detected: false
    });
  };

  const handleStartMonitoring = async () => {
    setErrorMessage(null); // Clear any previous errors
    resetMonitoringResults(); // Clear previous results
    setIsMonitoring(true);
    setMonitoringStatus('Initializing video processing...');

    let videoPathForBackend: string | null = null;
    
    let endpoint = "https://bd8b46660df3.ngrok-free.app/start";

    if (selectedSource === 'default') {
      videoPathForBackend = "vandalism.mp4"; // Backend expects just the filename if it's in the default video folder
    } else if (selectedSource === 'upload') {
      if (uploadedFile) {
        setMonitoringStatus('Uploading video file...');
        try {
          const formData = new FormData();
          formData.append('video', uploadedFile);
          
          const uploadResponse = await axios.post("https://bd8b46660df3.ngrok-free.app/upload_video", formData, {

            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          videoPathForBackend = uploadResponse.data.filepath; // Get the path from the backend response
          setMonitoringStatus('Video uploaded. Starting analysis...');
        } catch (uploadError: any) {
          console.error("Error uploading file:", uploadError);
          setErrorMessage(uploadError.response?.data?.error || "Failed to upload video file.");
          setMonitoringStatus('Upload failed.');
          setIsMonitoring(false);
          return;
        }
      } else {
        setErrorMessage("Please select a video file to upload.");
        setIsMonitoring(false);
        setMonitoringStatus('No file selected.');
        return;
      }
    }
    // If you add a 'webcam' option later:
    // else if (selectedSource === 'webcam') {
    //   videoPathForBackend = "webcam";
    // }

    try {
      await axios.post(endpoint, {
        video_path: videoPathForBackend,
        email_for_report: emailForReport,
      });
      setMonitoringStatus('Processing video stream...');
      setVideoFeedKey(Date.now()); // Refresh video feed to ensure it starts
    } catch (error: any) {
      console.error("Error starting monitoring:", error);
      setErrorMessage(error.response?.data?.error || "Failed to start monitoring. Check server connection.");
      setMonitoringStatus('Monitoring failed.');
      setIsMonitoring(false);
    }
  };

  const handleStopMonitoring = async () => {
    try {
      setMonitoringStatus('Stopping monitoring...');
      
      await axios.post("https://bd8b46660df3.ngrok-free.app/stop");

      setIsMonitoring(false);
      setMonitoringStatus('Monitoring stopped.');
      setVideoFeedKey(Date.now()); // Refresh video feed to clear stream
      // Do not reset results immediately here, keep them for review
    } catch (error: any) {
      console.error("Error stopping monitoring:", error);
      setErrorMessage(error.response?.data?.error || "Failed to stop monitoring.");
      setMonitoringStatus('Failed to stop monitoring.');
    }
  };

  return (
    <div className={`${darkMode ? 'dark' : ''} transition-all duration-500`}>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-100 dark:from-gray-900 dark:via-blue-900 dark:to-purple-900 relative overflow-hidden">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-400/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
          <div className="absolute top-1/2 left-1/2 w-60 h-60 bg-indigo-400/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        </div>

        {/* Top Navigation */}
        <header className="relative z-10 bg-white/70 dark:bg-gray-900/70 backdrop-blur-xl border-b border-gray-200/30 dark:border-gray-700/30 shadow-lg">
          <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl blur opacity-30 animate-pulse"></div>
                <div className="relative bg-gradient-to-r from-blue-500 to-purple-600 p-3 rounded-xl">
                  <Eye className="w-6 h-6 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  CrowdVision AI
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">Advanced Security Monitoring</p>
              </div>
            </div>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="group relative p-3 rounded-xl bg-gray-100/80 dark:bg-gray-800/80 hover:bg-gray-200/80 dark:hover:bg-gray-700/80 transition-all duration-300 hover:shadow-lg hover:scale-105"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-600/20 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="relative">
                {darkMode ? (
                  <Sun className="w-5 h-5 text-amber-500 transition-transform duration-300 group-hover:rotate-180" />
                ) : (
                  <Moon className="w-5 h-5 text-blue-600 transition-transform duration-300 group-hover:-rotate-12" />
                )}
              </div>
            </button>
          </div>
        </header>

        {/* Main Content */}
        <main className="relative z-10 max-w-7xl mx-auto p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Controls Panel */}
            <div className="lg:col-span-1">
              <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/30 dark:border-gray-700/30 p-8 space-y-8 hover:shadow-3xl transition-all duration-500">
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg blur opacity-30"></div>
                    <div className="relative bg-gradient-to-r from-emerald-500 to-blue-500 p-2 rounded-lg">
                      <Shield className="w-5 h-5 text-white" />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-200 bg-clip-text text-transparent">
                    Monitoring Controls
                  </h3>
                </div>

                {/* Video Source Selection */}
                <div className="space-y-4">
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">
                    Select Video Source
                  </label>
                  <div className="space-y-3">
                    {[
                      { value: 'default', label: 'Default Video (theft.mp4)', icon: Camera },
                      { value: 'upload', label: 'Upload Video', icon: Upload } // Renamed for clarity
                    ].map(({ value, label, icon: Icon }) => (
                      <label key={value} className="group relative block">
                        <input
                          type="radio"
                          className="sr-only"
                          name="videoSource"
                          value={value}
                          checked={selectedSource === value}
                          onChange={() => handleSourceChange(value)}
                        />
                        <div className={`
                          relative flex items-center p-4 rounded-xl border-2 cursor-pointer transition-all duration-300
                          ${selectedSource === value
                            ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-900/30 shadow-lg scale-[1.02]'
                            : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-gray-50/50 dark:hover:bg-gray-800/30'
                          }
                        `}>
                          <div className={`
                            p-2 rounded-lg mr-3 transition-all duration-300
                            ${selectedSource === value
                              ? 'bg-blue-500 text-white'
                              : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 group-hover:bg-blue-100 dark:group-hover:bg-blue-900/50'
                            }
                          `}>
                            <Icon className="w-4 h-4" />
                          </div>
                          <span className={`
                            text-sm font-medium transition-colors duration-300
                            ${selectedSource === value
                              ? 'text-blue-700 dark:text-blue-300'
                              : 'text-gray-700 dark:text-gray-300'
                            }
                          `}>
                            {label}
                          </span>
                          {selectedSource === value && (
                            <div className="absolute top-2 right-2">
                              <CheckCircle className="w-4 h-4 text-blue-500" />
                            </div>
                          )}
                        </div>
                      </label>
                    ))}
                  </div>

                  {selectedSource === 'upload' && (
                    <div className="mt-4 p-4 bg-gray-50/50 dark:bg-gray-800/50 rounded-xl border-2 border-dashed border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500 transition-all duration-300">
                      <input
                        type="file"
                        accept="video/*"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-gray-600 dark:text-gray-300
                          file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                          file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700
                          hover:file:bg-blue-100 dark:file:bg-blue-900/50 dark:file:text-blue-300
                          file:transition-all file:duration-300 file:hover:scale-105"
                      />
                      {uploadedFile && (
                        <p className="text-xs text-green-600 dark:text-green-400 mt-2 flex items-center">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          {uploadedFile.name} selected. Ready for upload.
                        </p>
                      )}
                    </div>
                  )}
                </div>

                {/* Email Input - Now only for sending to backend at start */}
                <div className="space-y-3">
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Email for Automated Report (optional)
                  </label>
                  <div className="flex relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 dark:text-gray-500" />
                    <input
                      type="email"
                      className="w-full pl-9 pr-4 py-3 rounded-xl border-2 border-gray-200 dark:border-gray-700
                        bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm
                        focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20
                        transition-all duration-300 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400
                        hover:border-gray-300 dark:hover:border-gray-600"
                      placeholder="your.email@example.com"
                      value={emailForReport}
                      onChange={(e) => setEmailForReport(e.target.value)}
                    />
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    A summary report will be automatically sent here upon video completion if alerts occur.
                  </p>
                </div>

                {/* Control Buttons */}
                <div className="space-y-4">
                  {!isMonitoring ? (
                    <button
                      onClick={handleStartMonitoring}
                      disabled={selectedSource === 'upload' && !uploadedFile}
                      className="group relative w-full py-4 px-6 rounded-xl bg-gradient-to-r from-emerald-500 to-blue-500
                        hover:from-emerald-600 hover:to-blue-600 text-white font-bold text-lg
                        disabled:opacity-50 disabled:cursor-not-allowed
                        transition-all duration-300 hover:shadow-2xl hover:scale-[1.02]
                        focus:ring-4 focus:ring-emerald-500/20 overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-blue-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                      <div className="relative flex items-center justify-center">
                        <Play className="mr-3 h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                        <span>Start Monitoring</span>
                      </div>
                    </button>
                  ) : (
                    <button
                      onClick={handleStopMonitoring}
                      className="group relative w-full py-4 px-6 rounded-xl bg-gradient-to-r from-red-500 to-red-600
                        hover:from-red-600 hover:to-red-700 text-white font-bold text-lg
                        transition-all duration-300 hover:shadow-2xl hover:scale-[1.02]
                        focus:ring-4 focus:ring-red-500/20 overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-red-400 to-red-500 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                      <div className="relative flex items-center justify-center">
                        <Square className="mr-3 h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                        <span>Stop Processing</span>
                      </div>
                    </button>
                  )}
                </div>

                {/* Status Display */}
                <div className="relative p-4 rounded-xl bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-700 border border-gray-200/50 dark:border-gray-600/50">
                  <div className="flex items-start space-x-3">
                    <div className={`
                      mt-0.5 w-3 h-3 rounded-full transition-all duration-500
                      ${isMonitoring ? 'bg-green-500 animate-pulse shadow-lg shadow-green-500/50' :
                        errorMessage ? 'bg-red-500 shadow-lg shadow-red-500/50' :
                        'bg-gray-400'}
                    `}></div>
                    <div className="flex-1">
                      <p className="font-semibold text-gray-800 dark:text-gray-200 text-sm mb-1">
                        System Status
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        {monitoringStatus}
                      </p>
                      {errorMessage && (
                        <div className="mt-2 p-3 bg-red-50 dark:bg-red-900/30 rounded-lg border border-red-200 dark:border-red-800">
                          <p className="text-red-600 dark:text-red-400 flex items-center text-sm">
                            <XCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                            {errorMessage}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Video Feed & Results */}
            <div className="lg:col-span-2">
              <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/30 dark:border-gray-700/30 overflow-hidden hover:shadow-3xl transition-all duration-500">
                {/* Video Header */}
                <div className="p-6 bg-gradient-to-r from-blue-50/50 to-purple-50/50 dark:from-blue-900/20 dark:to-purple-900/20 border-b border-gray-200/30 dark:border-gray-700/30">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg blur opacity-30"></div>
                        <div className="relative bg-gradient-to-r from-blue-500 to-purple-600 p-2 rounded-lg">
                          <Eye className="w-5 h-5 text-white" />
                        </div>
                      </div>
                      <div>
                        <h3 className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-200 bg-clip-text text-transparent">
                          Live Vision Feed
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          Real-time analysis and detection
                        </p>
                      </div>
                    </div>
                    {isMonitoring && (
                      <div className="flex items-center space-x-2 text-sm">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-red-500 font-medium">
                          ANALYZING
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Real-time Counts */}
                <div className="p-4 bg-gradient-to-r from-gray-50/80 to-blue-50/80 dark:from-gray-800/50 dark:to-blue-900/30">
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      { label: 'Current Crowd', value: monitoringResults.current_crowd_count, color: 'blue', icon: Users },
                      { label: 'Standing', value: monitoringResults.standing_detections, color: 'green', icon: Users },
                      { label: 'Bending', value: monitoringResults.bending_detections, color: 'amber', icon: UserX },
                      { label: 'Unknown Faces', value: monitoringResults.unknown_faces, color: 'red', icon: AlertTriangle }
                    ].map(({ label, value, color, icon: Icon }) => (
                      <div key={label} className="group relative">
                        <div className={`
                          p-4 rounded-xl bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border border-gray-200/50 dark:border-gray-600/50
                          hover:shadow-lg transition-all duration-300 hover:scale-105
                          ${color === 'blue' ? 'hover:border-blue-300 dark:hover:border-blue-600' :
                            color === 'green' ? 'hover:border-green-300 dark:hover:border-green-600' :
                            color === 'amber' ? 'hover:border-amber-300 dark:hover:border-amber-600' :
                            'hover:border-red-300 dark:hover:border-red-600'}
                        `}>
                          <div className="flex items-center justify-between mb-2">
                            <Icon className={`
                              w-4 h-4 transition-all duration-300 group-hover:scale-110
                              ${color === 'blue' ? 'text-blue-500' :
                                color === 'green' ? 'text-green-500' :
                                color === 'amber' ? 'text-amber-500' :
                                'text-red-500'}
                            `} />
                            {value > 0 && (
                              <div className={`
                                w-2 h-2 rounded-full animate-pulse
                                ${color === 'blue' ? 'bg-blue-500' :
                                  color === 'green' ? 'bg-green-500' :
                                  color === 'amber' ? 'bg-amber-500' :
                                  'bg-red-500'}
                              `}></div>
                            )}
                          </div>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-medium">
                            {label}
                          </p>
                          <p className={`
                            text-2xl font-bold transition-all duration-300
                            ${color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                              color === 'green' ? 'text-green-600 dark:text-green-400' :
                              color === 'amber' ? 'text-amber-600 dark:text-amber-400' :
                              'text-red-600 dark:text-red-400'}
                          `}>
                            {value}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Video Display */}
                <div className="relative bg-gradient-to-br from-gray-900 to-black">
                  <div className="flex items-center justify-center h-[60vh] relative overflow-hidden">
                    {selectedSource === 'upload' && videoPreviewUrl && (
                      // Display local video preview if selected for upload
                      <video
                        src={videoPreviewUrl}
                        controls
                        className="w-full h-full object-contain transition-all duration-500 hover:scale-105"
                      />
                    )}
                    {selectedSource === 'default' && (
                      <img
                        
                        src={`https://bd8b46660df3.ngrok-free.app/video_feed?${videoFeedKey}`} 

                        alt="Default Video"
                        className="w-full h-full object-contain transition-all duration-500 hover:scale-105"
                      />
                    )}
                    {!selectedSource && ( // This state is not reachable with current logic (default is always selected initially)
                      <div className="text-center text-gray-400 dark:text-gray-500">
                        <div className="relative mb-6">
                          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-600/20 rounded-full blur-xl"></div>
                          <Camera className="relative w-16 h-16 mx-auto text-gray-300 dark:text-gray-600" />
                        </div>
                        <h4 className="text-xl font-semibold mb-2">No Video Source Selected</h4>
                        <p className="text-sm">Choose a video source from the control panel to begin monitoring</p>
                      </div>
                    )}

                    {/* Monitoring Overlay */}
                    {isMonitoring && (
                      <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm rounded-xl px-4 py-2 flex items-center space-x-2">
                        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-white text-sm font-medium">ANALYZING</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Analysis Summary */}
                <div className="p-6 bg-gradient-to-r from-gray-50/80 to-blue-50/80 dark:from-gray-800/50 dark:to-blue-900/30">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg blur opacity-30"></div>
                      <div className="relative bg-gradient-to-r from-emerald-500 to-blue-500 p-2 rounded-lg">
                        <Activity className="w-5 h-5 text-white" />
                      </div>
                    </div>
                    <div>
                      <h4 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-200 bg-clip-text text-transparent">
                        Analysis Summary
                      </h4>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Comprehensive detection results
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Crowd Statistics */}
                    <div className="group relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-600/10 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      <div className="relative bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-2xl border border-gray-200/50 dark:border-gray-600/50 hover:shadow-lg transition-all duration-300">
                        <div className="flex items-center space-x-3 mb-4">
                          <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                            <Users className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                          </div>
                          <h5 className="font-semibold text-gray-800 dark:text-gray-200">Crowd Analytics</h5>
                        </div>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Maximum Count</span>
                            <span className="font-bold text-lg text-blue-600 dark:text-blue-400">
                              {monitoringResults.maximum_crowd_count}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Known Faces</span>
                            <span className="font-bold text-lg text-green-600 dark:text-green-400">
                              {monitoringResults.known_faces}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Unknown Faces</span>
                            <span className="font-bold text-lg text-red-600 dark:text-red-400">
                              {monitoringResults.unknown_faces}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Posture Detection */}
                    <div className="group relative">
                      <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      <div className="relative bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm p-6 rounded-2xl border border-gray-200/50 dark:border-gray-600/50 hover:shadow-lg transition-all duration-300">
                        <div className="flex items-center space-x-3 mb-4">
                          <div className="p-2 bg-emerald-100 dark:bg-emerald-900/50 rounded-lg">
                            <UserX className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                          </div>
                          <h5 className="font-semibold text-gray-800 dark:text-gray-200">Posture Detection</h5>
                        </div>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Standing Detections</span>
                            <span className="font-bold text-lg text-green-600 dark:text-green-400">
                              {monitoringResults.standing_detections}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Bending Detections</span>
                            <span className="font-bold text-lg text-amber-600 dark:text-amber-400">
                              {monitoringResults.bending_detections}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 dark:text-gray-400">Theft Detected</span>
                            <span className={`font-bold text-lg ${monitoringResults.theft_detected ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'}`}>
                              {monitoringResults.theft_detected ? 'YES' : 'NO'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
